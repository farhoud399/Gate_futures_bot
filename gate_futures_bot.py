\
    import os, time, math, sys
    from datetime import datetime, timezone
    from typing import List, Tuple

    import pandas as pd
    from dotenv import load_dotenv

    import gate_api
    from gate_api import FuturesApi, ApiClient, Configuration
    from gate_api.exceptions import GateApiException, ApiException
    from gate_api.models import FuturesOrder, FuturesPriceTriggeredOrder

    # ---------- utils ----------
    def ema(series: pd.Series, span: int) -> pd.Series:
        return series.ewm(span=span, adjust=False).mean()

    def rsi(series: pd.Series, length: int = 14) -> pd.Series:
        delta = series.diff()
        up = delta.clip(lower=0)
        down = -1 * delta.clip(upper=0)
        roll_up = up.ewm(alpha=1/length, adjust=False).mean()
        roll_down = down.ewm(alpha=1/length, adjust=False).mean()
        rs = roll_up / roll_down.replace(0, 1e-9)
        return 100 - (100 / (1 + rs))

    def load_env():
        load_dotenv()
        cfg = Configuration(
            key=os.getenv("GATE_API_KEY"),
            secret=os.getenv("GATE_API_SECRET"),
        )
        use_testnet = os.getenv("USE_TESTNET", "true").lower() == "true"
        # Futures endpoints can use fx-api domains for testnet
        if use_testnet:
            cfg.host = "https://fx-api-testnet.gateio.ws/api/v4"
        else:
            cfg.host = "https://api.gateio.ws/api/v4"
        client = ApiClient(cfg)
        fapi = FuturesApi(client)

        params = {
            "SETTLE": os.getenv("SETTLE", "usdt"),
            "CONTRACT": os.getenv("CONTRACT", "BTC_USDT"),
            "LEVERAGE": int(os.getenv("LEVERAGE", "10")),
            "MARGIN_MODE": os.getenv("MARGIN_MODE", "ISOLATED").upper(),
            "RISK_PCT": float(os.getenv("RISK_PCT", "0.1")),
            "INTERVAL": os.getenv("INTERVAL", "1m"),
            "FAST_EMA": int(os.getenv("FAST_EMA", "12")),
            "SLOW_EMA": int(os.getenv("SLOW_EMA", "26")),
            "RSI_LEN": int(os.getenv("RSI_LEN", "14")),
            "RSI_BUY": float(os.getenv("RSI_BUY", "35")),
            "RSI_SELL": float(os.getenv("RSI_SELL", "65")),
            "STOP_LOSS_PCT": float(os.getenv("STOP_LOSS_PCT", "0.005")),
            "TAKE_PROFIT_PCT": float(os.getenv("TAKE_PROFIT_PCT", "0.01")),
        }
        return fapi, params

    def get_contract_spec(fapi: FuturesApi, settle: str, contract: str) -> dict:
        spec = fapi.get_futures_contract(settle, contract)
        return spec.to_dict()

    def get_ticker(fapi: FuturesApi, settle: str, contract: str) -> dict:
        return fapi.get_futures_ticker(settle, contract).to_dict()

    def get_account(fapi: FuturesApi, settle: str) -> dict:
        return fapi.get_futures_account(settle).to_dict()

    def fetch_klines_df(fapi: FuturesApi, settle: str, contract: str, interval: str, limit: int = 300) -> pd.DataFrame:
        # API returns list of [t, o, h, l, c, v]
        kl = fapi.list_futures_candlesticks(settle, contract, limit=limit, interval=interval)
        rows = [[int(x[0]), float(x[1]), float(x[2]), float(x[3]), float(x[4]), float(x[5])] for x in kl]
        df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"])
        df["time"] = pd.to_datetime(df["ts"], unit="s", utc=True)
        return df

    def ensure_modes(fapi: FuturesApi, settle: str, contract: str, leverage: int, margin_mode: str):
        # set isolated / cross
        try:
            body = {"mode": margin_mode, "contract": contract}
            fapi.switch_position_margin_mode(settle, body)
        except GateApiException as e:
            print(f"[WARN] switch_position_margin_mode failed: {e}", file=sys.stderr)
        # set leverage
        try:
            fapi.update_position_leverage(settle, contract, str(leverage))
        except GateApiException as e:
            print(f"[WARN] update_position_leverage failed: {e}", file=sys.stderr)

    def compute_signal(df: pd.DataFrame, fast: int, slow: int, rsi_len: int, rsi_buy: float, rsi_sell: float) -> str:
        df = df.copy()
        df["ema_fast"] = ema(df["close"], fast)
        df["ema_slow"] = ema(df["close"], slow)
        df["rsi"] = rsi(df["close"], rsi_len)
        # Signals
        cross_up = df["ema_fast"].iloc[-2] < df["ema_slow"].iloc[-2] and df["ema_fast"].iloc[-1] > df["ema_slow"].iloc[-1]
        cross_dn = df["ema_fast"].iloc[-2] > df["ema_slow"].iloc[-2] and df["ema_fast"].iloc[-1] < df["ema_slow"].iloc[-1]
        if cross_up and df["rsi"].iloc[-1] <= rsi_sell:
            return "LONG"
        if cross_dn and df["rsi"].iloc[-1] >= rsi_buy:
            return "SHORT"
        return "FLAT"

    def calc_order_size_by_risk(available_usdt: float, price: float, multiplier: float, leverage: int, risk_pct: float) -> int:
        # Margin we aim to use per trade:
        margin_target = max(0.0, available_usdt) * max(0.0, min(risk_pct, 1.0))
        if margin_target <= 0:
            return 0
        notional = margin_target * leverage
        contracts = int(notional / (price * multiplier))
        return max(0, contracts)

    def place_market_order(fapi: FuturesApi, settle: str, contract: str, size: int, text: str = "") -> dict:
        # price=0 + tif='ioc' => market per Gate docs
        order = FuturesOrder(contract=contract, size=size, price="0", tif="ioc", text=text)
        return fapi.create_futures_order(settle, order).to_dict()

    def place_sl_tp(fapi: FuturesApi, settle: str, contract: str, size: int, entry_price: float, side: str, sl_pct: float, tp_pct: float):
        # Reduce-only stop & take profit using price-triggered orders on mark price
        reduce_size = abs(size)  # number of contracts to close
        if reduce_size == 0:
            return None, None
        if side.upper() == "LONG":
            sl_trigger = entry_price * (1 - sl_pct)
            tp_trigger = entry_price * (1 + tp_pct)
            close_side = "sell"
        else:
            sl_trigger = entry_price * (1 + sl_pct)
            tp_trigger = entry_price * (1 - tp_pct)
            close_side = "buy"
        # price order model
        def mk_price_order(trigger_price: float):
            return FuturesPriceTriggeredOrder(
                initial=FuturesPriceTriggeredOrder.Initial(
                    contract=contract,
                    size=reduce_size * (-1 if close_side == "sell" else 1),
                    price="0",  # market on trigger
                    tif="ioc",
                    reduce_only=True
                ),
                trigger=FuturesPriceTriggeredOrder.Trigger(
                    rule="mark_price",
                    price=round(trigger_price, 2)  # Gate tick size for BTCUSDT is 0.1; 2 decimals is usually safe
                )
            )
        try:
            sl = mk_price_order(sl_trigger)
            sl_res = fapi.create_futures_price_order(settle, sl).to_dict()
        except Exception as e:
            sl_res = {"error": str(e)}
        try:
            tp = mk_price_order(tp_trigger)
            tp_res = fapi.create_futures_price_order(settle, tp).to_dict()
        except Exception as e:
            tp_res = {"error": str(e)}
        return sl_res, tp_res

    def main():
        fapi, P = load_env()
        settle = P["SETTLE"]
        contract = P["CONTRACT"]
        # Setup
        ensure_modes(fapi, settle, contract, P["LEVERAGE"], P["MARGIN_MODE"])
        # Market data
        df = fetch_klines_df(fapi, settle, contract, P["INTERVAL"], limit=200)
        sig = compute_signal(df, P["FAST_EMA"], P["SLOW_EMA"], P["RSI_LEN"], P["RSI_BUY"], P["RSI_SELL"])
        tk = get_ticker(fapi, settle, contract)
        price = float(tk["last"])
        spec = get_contract_spec(fapi, settle, contract)
        multiplier = float(spec.get("quanto_multiplier") or spec.get("multiplier") or 0.0001)
        acct = get_account(fapi, settle)
        available = float(acct.get("available", 0) or acct.get("available_margin", 0) or 0)
        print(f"[{datetime.now(timezone.utc).isoformat()}] Price={price} avail={available} mode={P['MARGIN_MODE']} lev={P['LEVERAGE']} sig={sig}")
        if sig == "FLAT":
            print("No signal -> no trade.")
            return
        # sizing
        side = "buy" if sig == "LONG" else "sell"
        size = calc_order_size_by_risk(available, price, multiplier, P["LEVERAGE"], P["RISK_PCT"])
        if size == 0:
            print("Size=0 -> not placing order. Check balance and settings.")
            return
        if side == "sell":
            size = -size
        # place market order
        order = place_market_order(fapi, settle, contract, size, text="phone-bot")
        entry = float(order.get("avg_price") or price)
        print("Order result:", order)
        # place SL/TP reduce-only
        sl_res, tp_res = place_sl_tp(fapi, settle, contract, size, entry, "LONG" if side=="buy" else "SHORT", P["STOP_LOSS_PCT"], P["TAKE_PROFIT_PCT"])
        print("SL:", sl_res)
        print("TP:", tp_res)

    if __name__ == "__main__":
    try:
        loop_s = int(os.getenv("LOOP_SECONDS", "0"))
        if loop_s > 0:
            while True:
                try:
                    main()
                except (GateApiException, ApiException) as e:
                    print(f"[Gate API ERROR] {e}", file=sys.stderr)
                except Exception as ex:
                    print(f"[ERROR] {ex}", file=sys.stderr)
                time.sleep(loop_s)
        else:
            main()
    except (GateApiException, ApiException) as e:
        print(f"[Gate API ERROR] {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as ex:
        print(f"[ERROR] {ex}", file=sys.stderr)
        sys.exit(1)
