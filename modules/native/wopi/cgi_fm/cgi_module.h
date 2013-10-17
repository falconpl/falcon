/*
   FALCON - The Falcon Programming Language.
   FILE: cgi_module.h

   Standalone CGI module for Falcon WOPI
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Mon, 14 Oct 2013 00:16:14 +0200

   -------------------------------------------------------------------
   (C) Copyright 2013: the FALCON developers (see list in AUTHORS file)
*/
#ifndef _FALCON_WOPI_CGI_MODULE_H_
#define _FALCON_WOPI_CGI_MODULE_H_

#include <falcon/wopi/modulewopi.h>

namespace Falcon {

class VMContext;
class Process;
class Stream;

namespace WOPI {

class ModuleCGI: public ModuleWopi
{
public:
   ModuleCGI();
   virtual ~ModuleCGI();

   /** Override startup notification.
    * This will be used to switch the standard process output streams.
    */
   virtual void onStartupComplete( VMContext* ctx );
private:

   Stream* m_oldStdOut;
   Stream* m_oldStdErr;
   Process* m_process;
};

}
}

#endif /* CGI_MODULE_H_ */

/* end of cgi_module.h */
