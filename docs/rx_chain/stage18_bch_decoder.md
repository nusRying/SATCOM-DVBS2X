Stage 18 – BCH Check/Strip
--------------------------
Purpose  
Verify outer BCH code and remove parity bits (detect‑only implementation). EN 302 307‑1 §5.3.1, Tables 5a/5b, 6a/6b.

Interface (rx/_06_bch_decoding.py)  
`bch_check_and_strip(codeword_bits, fecframe, rate) -> (payload, meta)`

Inputs  
- `codeword_bits` : uint8 length Nbch from BCH_PARAMS[(fecframe, rate)].  
- `fecframe` : "normal"/"short".  
- `rate` : same as LDPC rate.

Outputs  
- `payload` : first Kbch bits (BBFRAME after descramble).  
- `meta` : Kbch, Nbch, t, corrected=False, errors=0.  
- Raises ValueError if syndrome ≠ 0 (no correction attempted).

Math  
- Parameters: (Kbch, Nbch, t) from Table 5a/5b.  
- Generator g(x): product of first t polynomials from Table 6a (normal, deg 16) or 6b (short, deg 14):  
    g(x) = Π_{i=1..t} g_i(x)  
- Bits → poly int (MSB first): c(x).  
- Syndrome: s(x) = c(x) mod g(x). Pass if s(x)=0.  
- Output = first Kbch bits (systematic part); parity length = Nbch−Kbch.

Complexity  
- Polynomial modulo in O(Nbch·deg(g)); negligible vs LDPC.

Limitation / TODO  
- No error location/correction; standard allows correcting up to t errors. Current behavior drops frame on any non‑zero syndrome.
