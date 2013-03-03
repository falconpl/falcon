#include <stdio.h>
#include <stdlib.h>
#include <cstring>
#include <vector>
#include <string>
#include <iostream>
#include <editline/readline.h>


template<typename T>
struct FreeGuard {
  T* p;
  explicit FreeGuard(T* p) : p(p) { }
  ~FreeGuard() { free(p); }
};


//rl_getc_function
static int my_getc(FILE*)
{
  return getchar();
}

int main(int argc, char **argv) 
{
  rl_getc_function = my_getc;
  char *buf;
  while(buf = readline("\n >> ")) 
  {
    FreeGuard<char> freeGuard(buf);
    //enable auto-complete
    rl_bind_key('\t',rl_complete);
    
    printf("cmd [%s]\n",buf);
    if (strcmp(buf,"quit")==0)
      break;
        
    if (buf[0]!=0)
      add_history(buf);
  }  
}
