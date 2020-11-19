#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "random.h"

double get_random(){
    return (2.0 * rand() / RAND_MAX) - 1.0;
}