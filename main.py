from __future__ import annotations

import csv
import datetime
import math
import random
import string

STEPS = 1000
MAX_LENGTH = 20
LEARNING_RATE = 0.01
BETA_1, BETA_2 = 0.85, 0.99
EPS_ADAM = 1e-8


def prepare_dataset() -> list[str]:
    dataset = []
    with open("SUB-IP-EST2025-ANNRNK.csv", encoding="utf-8") as f:
        # Skip the title/description.
        next(f)
        next(f)

        reader = csv.DictReader(f)
        for row in reader:
            city_name = row["Geographic Area"]
            if not city_name:
                continue

            for suffix in ("city", "CDP", "urban county", "town", "municipality", "County", "village", "borough"):
                city_name = city_name.replace(f" {suffix} (balance), ", ", ")
                city_name = city_name.replace(f" {suffix}, ", ", ")
            for gov_suffix in ("metropolitan", "metro", "consolidated", "unified"):
                city_name = city_name.replace(f" {gov_suffix} government (balance), ", ", ")
            city_name = city_name.replace(f" (balance), ", ", ")

            assert city_name != row["Geographic Area"] or "Carson City" in city_name or "Princeton" in city_name, f"{row['Geographic Area'] = } {city_name = }"

            city_name = city_name.lower()

            if len(city_name) <= MAX_LENGTH and set(city_name) <= set(string.ascii_lowercase + ", "):
                dataset.append(city_name)

    random.shuffle(dataset)
    return dataset


class Value:
    __slots__ = ("data", "grad", "_children", "_local_grads")

    def __init__(self, data: float, children: tuple[Value, ...] = (), local_grads: tuple[float, ...] = ()):
        self.data = data
        self.grad = 0
        self._children = children
        self._local_grads = local_grads

    def __add__(self, other: Value | float) -> Value:
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data + other.data, (self, other), (1, 1))

    def __mul__(self, other: Value | float) -> Value:
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data * other.data, (self, other), (other.data, self.data))

    def __pow__(self, power) -> Value:
        return Value(self.data ** power, (self,), (power * self.data ** (power - 1),))

    def log(self) -> Value:
        return Value(math.log(self.data), (self,), (1 / self.data,))

    def exp(self) -> Value:
        return Value(math.exp(self.data), (self,), (math.exp(self.data),))

    def relu(self) -> Value:
        return Value(max(self.data, 0), (self,), (1 if self.data > 0 else 0,))

    def __neg__(self):
        return self * -1

    def __radd__(self, other):
        return self + other

    def __sub__(self, other: Value | float) -> Value:
        return self + (-other)

    def __rsub__(self, other: Value | float) -> Value:
        return other + (-self)

    def __rmul__(self, other: Value) -> Value:
        return self * other

    def __truediv__(self, other: Value | float) -> Value:
        return self * other ** -1

    def __rtruediv__(self, other: Value) -> Value:
        return other * self ** -1

    def backward(self) -> None:
        topological_order: list[Value] = []
        visited = set()

        def topological_sort(node: Value):
            visited.add(node)
            for child in node._children:
                if child not in visited:
                    topological_sort(child)
            topological_order.append(node)

        topological_sort(self)

        self.grad = 1
        for node in reversed(topological_order):
            for child, local_grad in zip(node._children, node._local_grads, strict=True):
                child.grad += node.grad * local_grad


# This is matrix(cols, rows) but I like having the in -> out ordering more.
matrix = lambda n_in, n_out: [[Value(random.gauss(0, sigma=0.08)) for _ in range(n_in)] for _ in range(n_out)]
add_embeddings = lambda a, b: [x + y for x, y in zip(a, b, strict=True)]


def linear(row: list[Value], mat: list[list[Value]], div: float = 1) -> list[Value]:
    return [
        sum((v * m_v for v, m_v in zip(row, mat_row, strict=True)), start=Value(0)) / div
        for mat_row in mat
    ]


def softmax(logits: list[Value]) -> list[Value]:
    max_val = max(val.data for val in logits)
    exps = [(val - max_val).exp() for val in logits]
    total = sum(exps)
    return [e / total for e in exps]


def rmsnorm(x: list[Value]) -> list[Value]:
    ms = sum(xi * xi for xi in x) / len(x)
    scale = (ms + 1e-5) ** -0.5
    return [xi * scale for xi in x]


class Layer:
    __slots__ = ("n_embed", "n_head", "attn_wq", "attn_wk", "attn_wv", "attn_wo", "mlp_fc1", "mlp_fc2")

    def __init__(self, n_embed: int, n_head: int, n_mlp: int):
        self.n_embed = n_embed
        self.n_head = n_head

        self.attn_wq = matrix(n_embed, n_embed)
        self.attn_wk = matrix(n_embed, n_embed)
        self.attn_wv = matrix(n_embed, n_embed)
        self.attn_wo = matrix(n_embed, n_embed)
        self.mlp_fc1 = matrix(n_embed, n_mlp)
        self.mlp_fc2 = matrix(n_mlp, n_embed)

    @property
    def params(self) -> list[Value]:
        return [
            val
            for mat in (self.attn_wq, self.attn_wk, self.attn_wv, self.attn_wo, self.mlp_fc1, self.mlp_fc2)
            for row in mat
            for val in row
        ]

    def adjust(
            self, embedding: list[Value], keys: list[list[Value]], values: list[list[Value]]
    ) -> tuple[list[Value], list[list[Value]], list[list[Value]]]:
        emb = rmsnorm(embedding)

        head_dim = self.n_embed // self.n_head

        query = linear(emb, self.attn_wq)
        key = linear(emb, self.attn_wk)
        val = linear(emb, self.attn_wv)
        keys.append(key)
        values.append(val)

        attn_out = []
        for i_head in range(self.n_head):
            query_h = query[head_dim * i_head: head_dim * (i_head + 1)]
            keys_h = [key[head_dim * i_head: head_dim * (i_head + 1)] for key in keys]
            vals_h = [val[head_dim * i_head: head_dim * (i_head + 1)] for val in values]
            vals_h = [[vals_h[r][c] for r in range(len(values))] for c in range(head_dim)]

            attn_logits = linear(query_h, keys_h, div=math.sqrt(head_dim))
            attn_weights = softmax(attn_logits)
            head_out = linear(attn_weights, vals_h)
            attn_out.extend(head_out)

        attn_adjustment = linear(attn_out, self.attn_wo)
        embedding = add_embeddings(embedding, attn_adjustment)

        emb = rmsnorm(embedding)
        emb_mlp = linear(emb, self.mlp_fc1)
        emb_mlp = [val.relu() for val in emb_mlp]
        mlp_adjustment = linear(emb_mlp, self.mlp_fc2)

        return add_embeddings(embedding, mlp_adjustment), keys, values


class Model:
    def __init__(self, n_layer: int, n_embed: int, context_window: int, n_head: int, n_mlp: int, tokens: str):
        self.n_layer = n_layer
        self.n_embed = n_embed
        self.context_window = context_window
        self.n_head = n_head
        self.n_mlp = n_mlp
        self.tokens = tokens
        self.n_tokens = len(tokens) + 1

        self.tok_embed = matrix(self.n_tokens, n_embed)
        self.pos_embed = matrix(context_window, n_embed)
        self.layers = [Layer(n_embed=n_embed, n_head=n_head, n_mlp=n_mlp) for _ in range(n_layer)]
        self.lm_head = matrix(n_embed, self.n_tokens)

        params = [val for mat in (self.tok_embed, self.pos_embed, self.lm_head) for row in mat for val in row]
        layer_params = [param for layer in self.layers for param in layer.params]
        self.params = params + layer_params

        self.opt_m = [0 for _ in self.params]
        self.opt_v = [0 for _ in self.params]

    def infer(
            self, token_id: int, position: int, keys: list[list[list[Value]]], values: list[list[list[Value]]]
    ) -> tuple[list[Value], list[list[list[Value]]], list[list[list[Value]]]]:
        token_embedding = [self.tok_embed[i][token_id] for i in range(self.n_embed)]
        position_embedding = [self.pos_embed[i][position] for i in range(self.n_embed)]
        embedding = add_embeddings(token_embedding, position_embedding)
        embedding = rmsnorm(embedding)

        for i_layer, layer in enumerate(self.layers):
            embedding, keys[i_layer], values[i_layer] = layer.adjust(embedding, keys[i_layer], values[i_layer])

        logits = linear(embedding, self.lm_head)
        return logits, keys, values

    def generate(self) -> str:
        delimiter = len(self.tokens)
        output = ""
        token = delimiter
        keys, values = [[] for _ in self.layers], [[] for _ in self.layers]
        for pos in range(MAX_LENGTH):
            logits, keys, values = self.infer(token, pos, keys, values)
            probs = [val.data for val in softmax(logits)]
            token = random.choices(range(self.n_tokens), weights=probs)[0]
            if token == delimiter:
                break

            output += self.tokens[token]
        return output

    def train_step(self, data: str, step: int, n_steps: int):
        delimiter = len(self.tokens)
        tokens = [delimiter] + [self.tokens.index(c) for c in data] + [delimiter]
        for param in self.params:
            param.grad = 0

        keys, values = [[] for _ in self.layers], [[] for _ in self.layers]

        losses = []
        n = min(MAX_LENGTH, len(tokens) - 1)
        for i in range(n):
            logits, keys, values = self.infer(tokens[i], i, keys, values)
            probs = softmax(logits)

            target = tokens[i + 1]
            losses.append(-probs[target].log())
        loss = sum(losses, start=Value(0)) / len(losses)
        loss.backward()

        learning_rate = LEARNING_RATE * (n_steps - step) / n_steps
        for i, param in enumerate(self.params):
            self.opt_m[i] = BETA_1 * self.opt_m[i] + (1 - BETA_1) * param.grad
            self.opt_v[i] = BETA_2 * self.opt_v[i] + (1 - BETA_2) * param.grad ** 2
            m_hat = self.opt_m[i] / (1 - BETA_1 ** (step + 1))
            v_hat = self.opt_v[i] / (1 - BETA_2 ** (step + 1))

            param.data -= learning_rate * m_hat / (v_hat ** 0.5 + EPS_ADAM)

        return loss


def main():
    n_layer = 2
    n_embed = 64
    n_head = 16
    n_mlp = 4 * 64

    print(f"{n_layer = } {n_embed = } {MAX_LENGTH = } {n_head = } {n_mlp = }")
    print(f"{LEARNING_RATE = } {BETA_1 = } {BETA_2 = } {EPS_ADAM = }")

    dataset = prepare_dataset()
    tokens = ''.join(sorted(set(''.join(dataset))))

    print(f"{len(dataset) = } {len(tokens) = }")
    print("Sample docs:", *dataset[:5], sep='\n')

    model = Model(n_layer=n_layer, n_embed=n_embed, context_window=MAX_LENGTH, n_head=n_head, n_mlp=n_mlp, tokens=tokens)
    for step in range(STEPS):
        loss = model.train_step(dataset[step % len(dataset)], step, STEPS)
        print(f"[{datetime.datetime.now().isoformat()}] step {step + 1:4d} / {STEPS:4d} | loss {loss.data:.4f}")

    for _ in range(20):
        print(model.generate())


if __name__ == '__main__':
    main()
