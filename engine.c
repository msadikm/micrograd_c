#include <stdio.h>
#include <stdlib.h>

typedef struct Value {
  float data;
  float grad;
  void (*_backward)(struct Value *);
  struct Value **_prev;
  int _prev_count;
  char _op;
} Value;

Value *Value_create(float data, Value **_children, int _children_count,
                    char _op);

Value *Value_add(Value *self, Value *other);

Value *Value_substract(Value *self, Value *other);

Value *Value_multiply(Value *self, Value *other);

Value *Value_divide(Value *self, Value *other);

void Value_backward(Value *self);

int main() {
  float input_val1, input_val2;

  printf("Val1: ");
  scanf("%f", &input_val1);
  printf("Val2: ");
  scanf("%f", &input_val2);

  Value *val1 = Value_create(input_val1, NULL, 0, '\0');
  Value *val2 = Value_create(input_val2, NULL, 0, '\0');

  printf("------- Addition --------'\n'");
  Value *addition_result = Value_add(val1, val2);
  printf("Addition: %f\n", addition_result->data);
  Value_backward(addition_result);
  printf("Gradient of val1: %f\n", val1->grad);
  printf("Gradient of val2: %f\n\n", val2->grad);

  printf("---------Substitution --------\n");
  Value *substraction_result = Value_substract(val1, val2);
  printf("Substraction: %f\n", substraction_result->data);
  Value_backward(substraction_result);
  printf("Gradient of val1: %f\n", val1->grad);
  printf("Gradient of val2: %f\n\n", val2->grad);

  printf("--------- Multiplication -------'\n'");
  Value *multiplication_result = Value_multiply(val1, val2);
  printf("Multiplication : %f\n", multiplication_result->data);
  Value_backward(multiplication_result);
  printf("Gradient of val1: %f\n", val1->grad);
  printf("Gradient of val2: %f\n\n", val2->grad);

  printf("--------- Divide -------'\n'");
  Value *divide_result = Value_divide(val1, val2);
  printf("Divide : %f\n", divide_result->data);
  Value_backward(divide_result);
  printf("Gradient of val1: %f\n", val1->grad);
  printf("Gradient of val2: %f\n", val2->grad);

  free(val1);
  free(val2);
  free(addition_result);
  free(substraction_result);
  free(multiplication_result);
  free(divide_result);
  // free(result);

  return 0;
}

Value *Value_create(float data, Value **_children, int _children_count,
                    char _op) {
  Value *value = (Value *)malloc(sizeof(Value));
  value->data = data;
  value->grad = 0.0;
  value->_backward = NULL;
  value->_op = _op;

  if (_children != NULL) {
    value->_prev = malloc(_children_count * sizeof(Value *));
    for (int i = 0; i < _children_count; ++i) {
      value->_prev[i] = _children[i];
    }
    value->_prev_count = _children_count;
  } else {
    value->_prev = NULL;
    value->_prev_count = 0;
  }

  return value;
}

void Value_add_backward(Value *self) {
  if (self->_prev_count == 2) {
    self->_prev[0]->grad += self->grad;
    self->_prev[1]->grad += self->grad;
  }
}

Value *Value_add(Value *self, Value *other) {
  Value *out =
      Value_create(self->data + other->data, (Value *[]){self, other}, 2, '+');
  out->_backward = Value_add_backward;
  return out;
}

void Value_substract_backward(Value *self) {
  if (self->_prev_count == 2) {
    self->_prev[0]->grad -= self->grad;
    self->_prev[1]->grad -= self->grad;
  }
}

Value *Value_substract(Value *self, Value *other) {
  Value *out =
      Value_create(self->data - other->data, (Value *[]){self, other}, 2, '*');
  out->_backward = Value_substract_backward;
  return out;
}

void Value_multiply_backward(Value *self) {
  if (self->_prev_count == 2) {
    self->_prev[0]->grad += self->grad * self->_prev[1]->data;
    self->_prev[1]->grad += self->grad * self->_prev[0]->data;
  }
}

Value *Value_multiply(Value *self, Value *other) {
  Value *out =
      Value_create(self->data * other->data, (Value *[]){self, other}, 2, '*');
  out->_backward = Value_multiply_backward;
  return out;
}

void Value_divide_backward(Value *self) {
  if (self->_prev_count == 2) {
    self->_prev[0]->grad += self->grad / self->_prev[1]->data;
    self->_prev[1]->grad -= self->grad * self->_prev[0]->data /
                            (self->_prev[1]->data * self->_prev[1]->grad);
  }
}

Value *Value_divide(Value *self, Value *other) {
  if (other->data != 0) {
    Value *out = Value_create(self->data / other->data,
                              (Value *[]){self, other}, 2, '/');
    out->_backward = Value_divide_backward;
    return out;
  } else {
    // special Value object representing an error or undefined result
    Value *err_value = Value_create(0, NULL, 0, 'E'); // 'E' stands for error
    err_value->grad = 0.0 / 0.0;                      // This result in NaN
    return err_value;
  }
}

void Value_backward(Value *self) {
  if (self->_backward != NULL) {
    self->_backward(self);
  }

  if (self->_prev != NULL) {
    for (int i = 0; i < self->_prev_count; ++i) {
      Value_backward(self->_prev[i]);
    }
  }
}
