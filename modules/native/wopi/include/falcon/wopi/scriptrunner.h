/*
   FALCON - The Falcon Programming Language.
   FILE: scriptrunner.h

   Web Oriented Programming Interface.

   Utility to configure and run a WOPI-oriented script process.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Wed, 15 Jan 2014 12:36:01 +0100

   -------------------------------------------------------------------
   (C) Copyright 2014: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef _FALCON_SCRIPTRUNNER_H_
#define _FALCON_SCRIPTRUNNER_H_

#include <falcon/setup.h>
#include <falcon/wopi/wopi.h>

namespace Falcon {
class VMachine;
class Log;

namespace WOPI {

class ErrorHandler;
class Client;

/** Utility to configure and run a WOPI-oriented script process.
 *
 * This class encapsulates a mini-embedding for a web-oriented
 * script.
 *
 * Basically, it does the following things:
 * # reads the request, eventually indicating an error.
 * # configures load path and common encoding for the VM Process.
 * # creates a core and a WOPI module, injecting them in the process.
 * # configures process streams out of the WOPI client definition.
 * # loads the main module.
 * # launches it.
 * # flushes the WOPI reply associated with the client.
 *
 * The ownership of request and reply object passes to the VM/GC
 * serving the script, if the script is launched.
 */
class ScriptRunner
{
public:
   /** Creates a script runner with the given virtual machine.
    *
    * The runner doesn't own the machine.
    */
   ScriptRunner( const String& provider, VMachine* vm, Log* log, ErrorHandler* eh );
   ~ScriptRunner();

   void run( Client* client, const String& localScript );

   void textEncoding( const String& value ) { m_sTextEncoding = value; }
   void sourceEncoding( const String& value ) { m_sSourceEncoding = value; }
   void loadPath( const String& value ) { m_loadPath = value; }

   /** Template WOPI object used for configuration */
   Wopi& templateWopi() { return m_template; }

private:
   String m_provider;
   Wopi m_template;
   VMachine* m_vm;
   Log* m_log;
   ErrorHandler* m_eh;

   String m_sTextEncoding;
   String m_sSourceEncoding;
   String m_loadPath;
};

}
}

#endif /* _FALCON_SCRIPTRUNNER_H_ */

/* end of scriptrunner.h */
