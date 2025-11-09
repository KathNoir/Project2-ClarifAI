# Test Code Examples for Comment Generator

## Simple Function (Good for Testing)

```python
def calculate_sum(a, b):
    return a + b
```

## More Complex Function (Better Token Extraction)

```python
def find_max_value(numbers):
    if not numbers:
        return None
    max_val = numbers[0]
    for num in numbers:
        if num > max_val:
            max_val = num
    return max_val
```

## Class with Methods (Best for Testing)

```python
class Calculator:
    def __init__(self):
        self.history = []
    
    def add(self, a, b):
        result = a + b
        self.history.append(f"{a} + {b} = {result}")
        return result
    
    def subtract(self, a, b):
        result = a - b
        self.history.append(f"{a} - {b} = {result}")
        return result
    
    def get_history(self):
        return self.history
```

## Recursive Function

```python
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)
```

## Sorting Algorithm (Complex Example)

```python
def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr
```

## Data Processing (Real-World Example)

```python
def process_user_data(users):
    valid_users = []
    for user in users:
        if user.get('email') and user.get('age') >= 18:
            valid_users.append({
                'name': user['name'],
                'email': user['email']
            })
    return valid_users
```

