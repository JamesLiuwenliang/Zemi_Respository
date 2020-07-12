> ### PCA導出プロセス

### 1. 第一主成分を取得する

#### 1.1 ターゲット式の簡略化

`PPT`から見で、目標は分散を最大化することです。
$$
Var(x) = \frac{1}{m}\sum_{i=1}^{m}(x_i-\bar{x})^{2}
$$
![1594559722631](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\1594559722631.png)

だから、写真から見でわかりやすい、この赤い線を見つけたい。

最初のステップはデミーン（`demean`）と呼ばれています。操作はすべてのサンプルの平均をゼロにすることです。

![1594560170250](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\1594560170250.png)

だから今の目標は
$$
Var(x) = \frac{1}{m}\sum_{i=1}^{m}(x_i)^{2}  ,  \bar{x} =0
$$
すべてのサンプルをデミーンしで、今求めているのはベクトル$\vec{w}$だと考えることができます。このベクトルは$\vec{ｗ}=(w1,w2)$で表示されることができます。また、このベクトルは単位ベクトルであると仮定します。

---

![1594565223839](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\1594565223839.png)

すべてのサンプル$X^{(i)}$が$\vec{w}$にマッピングされた後、次のように記述できます。
$$
Var(X_{project}) = \frac{1}{m}\sum_{i=1}^{i}(X_{project}^{(i)}-\bar{X}_{project})^{2}
$$
簡単な説明：

- $X_{project}$はベクトル$\vec{w}$にマッピングされる各ポイントです
- $Var(X_{project})$はマッピング後のこれらのポイントの分散です

**デミーン操作の後で、実際に、$X$は一つずつベクトルと考えることができます。**

ならば、このように変更できます：
$$
Var(X_{project}) = \frac{1}{m}\sum_{i=1}^{i}\left \|(X_{project}^{(i)})\right \|^{2}
$$
デミーン操作によって、 $\bar{X}_{project}=0$式が成り立つ
$$
Var(X_{project}) = \frac{1}{m}\sum_{i=1}^{i}\left \|(X_{project}^{(i)})\right \|^{2}
$$

---

![1594565231386](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\1594565231386.png)

ベクトル内積の定義によると
$$
X^{(i)}\cdot w =  \left \| X^{(i)} \right \| \cdot \left \| w \right \| \cdot cos\theta
$$
$\vec{w}$は単位ベクトルです
$$
X^{(i)}\cdot w =  \left \| X^{(i)} \right \| \cdot  cos\theta
$$
そしで、写真からわかりやすい式、$\left \| X^{(i)} \right \| \cdot   \cos\theta = \left \| X^{(i)}_project \right \|  $が成り立つ。今の目標は

$Var(X_{project}) = \frac{1}{m}\sum_{i=1}^{i}\left \|X^{(i)}\cdot w \right \|^{2}$最大化。

式の拡張：
$$
Var(X_{project}) = \frac{1}{m}\sum_{i=1}^{m}(X^{(i)}_{1}\cdot w_{1}  + X^{(i)}_{2}\cdot w_{2}+... X^{(i)}_{n}\cdot w_{n}  )^{2}
$$

$$
Var(X_{project}) = \frac{1}{m}\sum_{i=1}^{m}(\sum_{j=1}^{n} X^{(i)}_{j} w_{j}  )^{2}勾配上昇方法
$$

#### 1.2勾配上昇方法

今の目標は、$f(x)=\frac{1}{m}\sum_{i=1}^{m}(X^{(i)}_{1}\cdot w_{1}  + X^{(i)}_{2}\cdot w_{2}+... X^{(i)}_{n}\cdot w_{n}  )^{2}$最大化の時、$ｗ$を求める。そんなの問題は勾配上昇方法を使用することができます。
$$
\bigtriangledown f = \begin{bmatrix}
\frac{\partial f}{\partial w_1}\\ 
...\\ 
\frac{\partial f}{\partial w_n}
\end{bmatrix}=\frac{2}{m}\begin{bmatrix}
\sum_{i=1}^{m}(X^{(i)}_{1}\cdot w_{1}  + X^{(i)}_{2}\cdot w_{2}+... X^{(i)}_{n}\cdot w_{n} )X^{(i)}_1 \\ 
...\\ 
\sum_{i=1}^{m}(X^{(i)}_{1}\cdot w_{1}  + X^{(i)}_{2}\cdot w_{2}+... X^{(i)}_{n}\cdot w_{n} )X^{(i)}_n 
\end{bmatrix}
$$

$$
=\frac{2}{m}
\begin{bmatrix}
\sum_{i=1}^{m}(X^{(i)}w)X^{(i)}_1\\ 
...\\ 
\sum_{i=1}^{m}(X^{(i)}w)X^{(i)}_n
\end{bmatrix}
$$

$$
=\frac{2}{m} \cdot (
\begin{bmatrix}
X^{(1)}w & X^{(2)}w &...  & X^{(m)}w
\end{bmatrix} \cdot
\begin{bmatrix}
X^{(1)}_1 & X^{(1)}_2 &...  &X^{(1)}_n \\ 
X^{(2)}_1 & X^{(2)}_2 &...  &X^{(2)}_n \\ 
 ...& ... &...  &... \\ 
X^{(m)}_1 &  X^{(m)}_2&...  & X^{(m)}_n
\end{bmatrix} )^{T} =\frac{2}{m} \cdot X^{T}(Xw)
$$

以上です。

Pythonのコード：

```python
def f(w, X):
    return np.sum((X.dot(w)**2)) / len(X)

def df(w, X):
    return X.T.dot(X.dot(w)) * 2. / len(X)

def direction(w):
    return w / np.linalg.norm(w)

def first_component(X, initial_w, eta, n_iters = 1e4, epsilon=1e-8):
    
    w = direction(initial_w) 
    cur_iter = 0

    while cur_iter < n_iters:
        gradient = df(w, X)
        last_w = w
        w = w + eta * gradient
        w = direction(w) 
        if(abs(f(w, X) - f(last_w, X)) < epsilon):
            break
            
        cur_iter += 1

    return w
```

### 2. 第二主成分を取得する

第二主成分は、元のポイントが`w`軸にマッピングされた後の残差ベクトル部分です。

![1594569671478](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\1594569671478.png)

緑のベクトルは第二主成分です。
$$
{X}'^{(i)} = X^{i}-X^{i}_{project}
$$
Pythonのコード：

```python
# Xは最初の全部データのマトリックス，wは第一主成分のベクトル
X2 = X - X.dot(w).reshape(-1, 1) * w
```

### ３. 高次元データの低次元データへのマッピング

例えば、全部のデータセットはｍ個サンプルとｎ個特徴
$$
X= \begin{bmatrix}
X_1^{(1)} & X_2^{(1)} & ... &  X_n^{(1)}\\
X_1^{(2)} & X_2^{(2)} &...  & X_n^{(2)}\\
 ...& ... & ... & ...\\ 
 X_1^{(m)}& X_2^{(m)} & ... & X_n^{(m)}
\end{bmatrix}
$$
最初のk個の主成分のデータを見つけたい。
$$
W_k= \begin{bmatrix}
W_1^{(1)} & W_2^{(1)} & ... &  W_n^{(1)}\\
W_1^{(2)} & W_2^{(2)} &...  & W_n^{(2)}\\
 ...& ... & ... & ...\\ 
W_1^{(k)}& W_2^{(k)} & ... & W_n^{(k)}
\end{bmatrix}
$$
これらの2つの行列を使用して、次元削減操作を実行できし、次元数が少ない新しいデータセットを取得できます。
$$
X_k = X \cdot W_k^{T}
$$
元の次元数に戻したい場合は、操作を逆にすることができます。
$$
X_m = X_k \cdot W_k
$$
**しかし、このステップは情報の損失を伴う必要があります。**

#### 4. PCAアルゴリズムのコード

```python
import numpy as np


class PCA:

    def __init__(self, n_components):
        assert n_components >= 1, "n_components must be valid"
        self.n_components = n_components
        self.components_ = None

    def fit(self, X, eta=0.01, n_iters=1e4):
        assert self.n_components <= X.shape[1], \
            "n_components must not be greater than the feature number of X"

        def demean(X):
            return X - np.mean(X, axis=0)

        def f(w, X):
            return np.sum((X.dot(w) ** 2)) / len(X)

        def df(w, X):
            return X.T.dot(X.dot(w)) * 2. / len(X)

        def direction(w):
            return w / np.linalg.norm(w)

        def first_component(X, initial_w, eta=0.01, n_iters=1e4, epsilon=1e-8):

            w = direction(initial_w)
            cur_iter = 0

            while cur_iter < n_iters:
                gradient = df(w, X)
                last_w = w
                w = w + eta * gradient
                w = direction(w)
                if (abs(f(w, X) - f(last_w, X)) < epsilon):
                    break

                cur_iter += 1

            return w

        X_pca = demean(X)
        self.components_ = np.empty(shape=(self.n_components, X.shape[1]))
        for i in range(self.n_components):
            initial_w = np.random.random(X_pca.shape[1])
            w = first_component(X_pca, initial_w, eta, n_iters)
            self.components_[i,:] = w

            X_pca = X_pca - X_pca.dot(w).reshape(-1, 1) * w

        return self

    def transform(self, X):
        assert X.shape[1] == self.components_.shape[1]

        return X.dot(self.components_.T)

    def inverse_transform(self, X):
        assert X.shape[1] == self.components_.shape[0]

        return X.dot(self.components_)

    def __repr__(self):
        return "PCA(n_components=%d)" % self.n_components
```

















