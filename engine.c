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

Value *Value_create(float data, Value **_children, int _children_count, char _op);

Value *Value_add(Value *self, Value *other);

void Value_backward(Value *self);

int main() {
  Value *val1 = Value_create(3.0, NULL, 0, '\0');
  Value *val2 = Value_create(4.0, NULL, 0, '\0');

  printf("------- Addition --------'\n'");

  Value *result = Value_add(val1, val2);

  // printf("Address of val1: %p\n", (void *)val1);
  // printf("Address of val2: %p\n", (void *)val2);
  // printf("Address of result: %p\n", (void *)result);

  printf("Result: %f\n", result->data);

  Value_backward(result);

  printf("Gradient of val1: %f\n", val1->grad);
  printf("Gradient of val2: %f\n", val2->grad);

  free(val1);
  free(val2);
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
		value->_prev = malloc(_children_count * sizeof(Value*));
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
