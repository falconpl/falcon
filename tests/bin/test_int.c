
#include <falcon/setup.h>
#include <stdio.h>

void FALCON_DYN_SYM int_call( int a, int b, int c )
{
   printf( "int_call( %d, %d, %d )\n", a, b, c );
}
