/*
   FALCON - The Falcon Programming Language.
   FILE: modulewopi.h

   Web Oriented Programming Interface main module
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Fri, 04 Oct 2013 14:34:08 +0200

   -------------------------------------------------------------------
   (C) Copyright 2013: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/


#ifndef FALCON_WOPI_MODULEWOPI_H
#define FALCON_WOPI_MODULEWOPI_H

#include <falcon/module.h>

namespace Falcon {
class Process;

namespace WOPI {

class ClassWOPI;
class Wopi;
class Request;
class Reply;

/**
   Specific WOPI interface Falcon Module
*/
class ModuleWopi: public Module
{
public:
   ModuleWopi( const String& name );
   virtual ~ModuleWopi();

   ClassWOPI* wopiClass() const { return m_classwopi; }
   Wopi* wopi() const { return m_wopi; }
   const String& scriptName() const { return m_scriptName; }
   const String& scriptPath() const { return m_scriptPath; }

   void scriptName( const String& value ) { m_scriptName = value; }
   void scriptPath( const String& value ) { m_scriptPath = value; }

protected:

   void interceptOutputStreams( Process* prc );
   void resumeOutputStreams();

private:
   ClassWOPI* m_classwopi;

   Wopi* m_wopi;
   Request* m_request;
   Reply* m_reply;

   String m_provider;

   Process* m_process;
   Stream* m_oldStdout;
   Stream* m_oldStderr;

   String m_scriptName;
   String m_scriptPath;
};

}
}

#endif

/* modulewopi.h */
