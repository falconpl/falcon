#include <stdio.h>
#include <stdlib.h>
#include <cstring>
#include <vector>
#include <string>
#include <editline/readline.h>

static char** my_completion(const char*, int ,int);
char* my_generator(const char*,int);

template<typename T>
struct FreeGuard {
  T* p;
  explicit FreeGuard(T* p) : p(p) { }
  ~FreeGuard() { free(p); }
};

using namespace std;


vector<string> cmds;


int main()
{
  char const* cmd [] = { "hello", "world", "hell" ,"word", "quit", " "};
  cmds.assign(cmd , cmd + sizeof(cmd)/sizeof(cmd[0]) );
  
  char *buf;

  rl_attempted_completion_function = my_completion;

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
    
  return 0;
}

static int abort_rl(int,int)
{ return _rl_abort_internal(); }

static char** my_completion( const char * text , int start,  int end)
{
  char** matches = 0;
  
  if (start == 0)
    matches = rl_completion_matches (text, &my_generator);
  else
    rl_bind_key('\t',abort_rl);

   return matches;
}

char* my_generator(const char* text, int state)
{
  static int list_index, len;
  char* ret = 0;

  if (!state) 
  {
    list_index = 0;
    len = strlen (text);
  }
  
  while(list_index < cmds.size()) 
  { 
    string const& cmd = cmds[list_index++];
    if (cmd.compare(0,len,text) == 0)
    {
      ret = strdup(cmd.c_str());
      break;
    }
  }
    
  return ret;
}
