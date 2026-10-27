"""
bch_codec.py

Código corretor de erros BCH para a marca de 30 bits, SEM mudar a rede:
os 30 bits escondidos na imagem passam a ser

    [ 15 bits de mensagem | 15 bits de paridade ]

e na extração até 3 bits errados (em qualquer posição) são corrigidos.

Usa o BCH(31, 16) binário (m=5, t=3) do bchlib, "encurtado" para (30, 15):
o 16º bit de mensagem é sempre 0, não vai para a imagem e é reposto na
decodificação. Se o decodificador tentar "corrigir" esse bit fixo, o erro
foi além da capacidade do código e a leitura é marcada como falha.

Uso:
    codec = BCH3015()
    bits30 = codec.encode(msg15)          # msg15: array 0/1 de 15 posições
    msg15_lida, n_corrigidos = codec.decode(bits30_extraidos)
    # msg15_lida = None quando há mais erros do que o código corrige
    # (detectado); se ele "corrigir" para outra mensagem, é um erro silencioso.
"""

import bchlib
import numpy as np


class BCH3015:
    DATA_BITS = 15    # bits de mensagem (informação)
    CODE_BITS = 30    # bits escondidos na imagem
    T = 3             # erros corrigidos

    def __init__(self):
        self.bch = bchlib.BCH(self.T, m=5)          # BCH(31, 16), corrige 3 erros
        assert self.bch.n == 31 and self.bch.ecc_bits == 15

    @staticmethod
    def _to_bytes(bits, n_bytes):
        padded = np.zeros(n_bytes * 8, dtype=np.uint8)
        padded[:len(bits)] = bits
        return bytearray(np.packbits(padded).tobytes())

    @staticmethod
    def _to_bits(data, n_bits):
        return np.unpackbits(np.frombuffer(bytes(data), dtype=np.uint8))[:n_bits]

    def encode(self, msg_bits):
        """15 bits de mensagem -> 30 bits para esconder (mensagem + paridade)."""
        msg_bits = np.asarray(msg_bits, dtype=np.uint8).reshape(-1)
        assert msg_bits.size == self.DATA_BITS
        data = self._to_bytes(msg_bits, 2)                   # 15 bits + 1 bit fixo 0
        ecc = self.bch.encode(bytes(data))
        parity = self._to_bits(ecc, self.bch.ecc_bits)       # 15 bits de paridade
        return np.concatenate([msg_bits, parity]).astype(np.uint8)

    def decode(self, code_bits):
        """
        30 bits extraídos -> (15 bits de mensagem, nº de bits corrigidos).
        Devolve (None, -1) quando há mais erros do que o código corrige.
        """
        code_bits = np.asarray(code_bits, dtype=np.uint8).reshape(-1)
        assert code_bits.size == self.CODE_BITS
        data = self._to_bytes(code_bits[:self.DATA_BITS], 2)
        ecc = self._to_bytes(code_bits[self.DATA_BITS:], 2)

        n_err = self.bch.decode(bytes(data), bytes(ecc))
        if n_err < 0:
            return None, -1
        if n_err > 0:
            # bits que não vão para a imagem: o 16º bit de mensagem (fixo em 0)
            # e o 16º bit de paridade (sobra do byte). O bchlib numera os bits
            # do fim para o começo de cada byte, então eles são as posições 8
            # e 24. "Corrigir" um deles significa que o erro passou do limite.
            if any(loc in (8, 24) for loc in self.bch.errloc[:n_err]):
                return None, -1
            self.bch.correct(data, ecc)
        return self._to_bits(data, self.DATA_BITS), int(n_err)


def bits_to_int(bits):
    """15 bits -> número (o ID guardado na marca)."""
    return int("".join(str(int(b)) for b in bits), 2)


def int_to_bits(value, n_bits=15):
    return np.array([int(c) for c in format(value, f"0{n_bits}b")], dtype=np.uint8)
