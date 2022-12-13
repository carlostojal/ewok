#include <ewok/raycast_ring_buffer.h>

void moveOnAxis(ewok::RaycastRingBuffer::Vector3i direction, int axis) {
    
    if (direction[axis] != 0) {

        int slice;

        if (direction[axis] > 0) {
            offset_[axis]++;
            slice = offset_[axis] + _N - 1;
        } else {
            offset_[axis]--;
            slice = offset_[axis];
        }

        switch (axis) {
            case 0:context->setXSlice(slice, empty_element_);
            break;
            case 1:context->setYSlice(slice, empty_element_);
            break;
            case 2:context->setZSlice(slice, empty_element_);
            break;

        }

    }
  }