// generated from rosidl_generator_c/resource/idl__functions.c.em
// with input from docs_turtlesim:msg/Vector3.idl
// generated code does not contain a copyright notice
#include "docs_turtlesim/msg/detail/vector3__functions.h"

#include <assert.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#include "rcutils/allocator.h"


bool
docs_turtlesim__msg__Vector3__init(docs_turtlesim__msg__Vector3 * msg)
{
  if (!msg) {
    return false;
  }
  // x
  // y
  // z
  return true;
}

void
docs_turtlesim__msg__Vector3__fini(docs_turtlesim__msg__Vector3 * msg)
{
  if (!msg) {
    return;
  }
  // x
  // y
  // z
}

bool
docs_turtlesim__msg__Vector3__are_equal(const docs_turtlesim__msg__Vector3 * lhs, const docs_turtlesim__msg__Vector3 * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  // x
  if (lhs->x != rhs->x) {
    return false;
  }
  // y
  if (lhs->y != rhs->y) {
    return false;
  }
  // z
  if (lhs->z != rhs->z) {
    return false;
  }
  return true;
}

bool
docs_turtlesim__msg__Vector3__copy(
  const docs_turtlesim__msg__Vector3 * input,
  docs_turtlesim__msg__Vector3 * output)
{
  if (!input || !output) {
    return false;
  }
  // x
  output->x = input->x;
  // y
  output->y = input->y;
  // z
  output->z = input->z;
  return true;
}

docs_turtlesim__msg__Vector3 *
docs_turtlesim__msg__Vector3__create()
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  docs_turtlesim__msg__Vector3 * msg = (docs_turtlesim__msg__Vector3 *)allocator.allocate(sizeof(docs_turtlesim__msg__Vector3), allocator.state);
  if (!msg) {
    return NULL;
  }
  memset(msg, 0, sizeof(docs_turtlesim__msg__Vector3));
  bool success = docs_turtlesim__msg__Vector3__init(msg);
  if (!success) {
    allocator.deallocate(msg, allocator.state);
    return NULL;
  }
  return msg;
}

void
docs_turtlesim__msg__Vector3__destroy(docs_turtlesim__msg__Vector3 * msg)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (msg) {
    docs_turtlesim__msg__Vector3__fini(msg);
  }
  allocator.deallocate(msg, allocator.state);
}


bool
docs_turtlesim__msg__Vector3__Sequence__init(docs_turtlesim__msg__Vector3__Sequence * array, size_t size)
{
  if (!array) {
    return false;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  docs_turtlesim__msg__Vector3 * data = NULL;

  if (size) {
    data = (docs_turtlesim__msg__Vector3 *)allocator.zero_allocate(size, sizeof(docs_turtlesim__msg__Vector3), allocator.state);
    if (!data) {
      return false;
    }
    // initialize all array elements
    size_t i;
    for (i = 0; i < size; ++i) {
      bool success = docs_turtlesim__msg__Vector3__init(&data[i]);
      if (!success) {
        break;
      }
    }
    if (i < size) {
      // if initialization failed finalize the already initialized array elements
      for (; i > 0; --i) {
        docs_turtlesim__msg__Vector3__fini(&data[i - 1]);
      }
      allocator.deallocate(data, allocator.state);
      return false;
    }
  }
  array->data = data;
  array->size = size;
  array->capacity = size;
  return true;
}

void
docs_turtlesim__msg__Vector3__Sequence__fini(docs_turtlesim__msg__Vector3__Sequence * array)
{
  if (!array) {
    return;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();

  if (array->data) {
    // ensure that data and capacity values are consistent
    assert(array->capacity > 0);
    // finalize all array elements
    for (size_t i = 0; i < array->capacity; ++i) {
      docs_turtlesim__msg__Vector3__fini(&array->data[i]);
    }
    allocator.deallocate(array->data, allocator.state);
    array->data = NULL;
    array->size = 0;
    array->capacity = 0;
  } else {
    // ensure that data, size, and capacity values are consistent
    assert(0 == array->size);
    assert(0 == array->capacity);
  }
}

docs_turtlesim__msg__Vector3__Sequence *
docs_turtlesim__msg__Vector3__Sequence__create(size_t size)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  docs_turtlesim__msg__Vector3__Sequence * array = (docs_turtlesim__msg__Vector3__Sequence *)allocator.allocate(sizeof(docs_turtlesim__msg__Vector3__Sequence), allocator.state);
  if (!array) {
    return NULL;
  }
  bool success = docs_turtlesim__msg__Vector3__Sequence__init(array, size);
  if (!success) {
    allocator.deallocate(array, allocator.state);
    return NULL;
  }
  return array;
}

void
docs_turtlesim__msg__Vector3__Sequence__destroy(docs_turtlesim__msg__Vector3__Sequence * array)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (array) {
    docs_turtlesim__msg__Vector3__Sequence__fini(array);
  }
  allocator.deallocate(array, allocator.state);
}

bool
docs_turtlesim__msg__Vector3__Sequence__are_equal(const docs_turtlesim__msg__Vector3__Sequence * lhs, const docs_turtlesim__msg__Vector3__Sequence * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  if (lhs->size != rhs->size) {
    return false;
  }
  for (size_t i = 0; i < lhs->size; ++i) {
    if (!docs_turtlesim__msg__Vector3__are_equal(&(lhs->data[i]), &(rhs->data[i]))) {
      return false;
    }
  }
  return true;
}

bool
docs_turtlesim__msg__Vector3__Sequence__copy(
  const docs_turtlesim__msg__Vector3__Sequence * input,
  docs_turtlesim__msg__Vector3__Sequence * output)
{
  if (!input || !output) {
    return false;
  }
  if (output->capacity < input->size) {
    const size_t allocation_size =
      input->size * sizeof(docs_turtlesim__msg__Vector3);
    rcutils_allocator_t allocator = rcutils_get_default_allocator();
    docs_turtlesim__msg__Vector3 * data =
      (docs_turtlesim__msg__Vector3 *)allocator.reallocate(
      output->data, allocation_size, allocator.state);
    if (!data) {
      return false;
    }
    // If reallocation succeeded, memory may or may not have been moved
    // to fulfill the allocation request, invalidating output->data.
    output->data = data;
    for (size_t i = output->capacity; i < input->size; ++i) {
      if (!docs_turtlesim__msg__Vector3__init(&output->data[i])) {
        // If initialization of any new item fails, roll back
        // all previously initialized items. Existing items
        // in output are to be left unmodified.
        for (; i-- > output->capacity; ) {
          docs_turtlesim__msg__Vector3__fini(&output->data[i]);
        }
        return false;
      }
    }
    output->capacity = input->size;
  }
  output->size = input->size;
  for (size_t i = 0; i < input->size; ++i) {
    if (!docs_turtlesim__msg__Vector3__copy(
        &(input->data[i]), &(output->data[i])))
    {
      return false;
    }
  }
  return true;
}
