class ExponentialSmoother:
    """Экспоненциальное сглаживание координат"""
    def __init__(self, alpha=0.85):
        self.alpha = alpha
        self.value = None

    def update(self, new_value):
        if self.value is None:
            self.value = new_value
        else:
            self.value = self.alpha * self.value + (1 - self.alpha) * new_value
        return self.value
