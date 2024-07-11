# numpy

- (broadcast ->) element-wise な計算

- 要素の取得:

```python
>>> x
array([11, 22, 33, 44, 55, 66])
>>> y
array([1, 3, 4])
>>> z  # z = x > 30
array([False, False,  True,  True,  True,  True])
>>> q
array([1, 1, 1, 0, 0, 1])
>>>
>>> x[y]
array([22, 44, 55])
>>> x[z]
array([33, 44, 55, 66])
>>> x[q]
array([22, 22, 22, 11, 11, 22])
```
