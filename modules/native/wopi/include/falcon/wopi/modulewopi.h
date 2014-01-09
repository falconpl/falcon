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

class ClassWopi;
class ClassRequest;
class ClassReply;
class ClassUploaded;
class Wopi;
class Request;
class Reply;

/**
   Specific WOPI interface Falcon Module.
*/
class ModuleWopi: public Module
{
public:
   /** Creates the module.
    * \param name Module name
    * \param req Eventually pre-parsed request.
    *
    * A WOPI request might be prepared before the module is
    * actually created, i.e. by parsing an incoming request
    * from the net and checking its validity.
    *
    * If a Request entity is not provided in advance, this
    * class creates one, else an empty request is created on
    * the fly and can be accessed via the request() member.
    */
   ModuleWopi( const String& name, Request* req = 0, Reply* rep = 0 );
   virtual ~ModuleWopi();

   ClassWopi* wopiClass() const { return m_classWopi; }
   ClassUploaded* uploadedClass() const { return m_classUploaded; }
   Wopi* wopi() const { return m_wopi; }
   const String& scriptName() const { return m_scriptName; }
   const String& scriptPath() const { return m_scriptPath; }

   void scriptName( const String& value ) { m_scriptName = value; }
   void scriptPath( const String& value ) { m_scriptPath = value; }

   const String& provider() const { return m_provider; }

   Request* request() const { return m_request; }
   ClassRequest* requestClass() const { return m_classRequest; }
   ClassReply* replyClass() const { return m_classReply; }
   Reply* reply() const { return m_reply; }

   Class* sessionClass() const { return m_sessionClass; }
   Module* sessionModule() const { return m_sessionModule; }

   /** Records the session class as it's resolved. */
   virtual void onModuleResolved( ModRequest* mr );
   virtual void onLinkComplete(VMContext*);

protected:

   void interceptOutputStreams( Process* prc );
   void resumeOutputStreams();

   ClassWopi* m_classWopi;
   ClassRequest* m_classRequest;
   ClassReply* m_classReply;
   ClassUploaded* m_classUploaded;

   Class* m_sessionClass;

   Wopi* m_wopi;
   Request* m_request;
   Reply* m_reply;

   String m_provider;

   Process* m_process;
   Stream* m_oldStdout;
   Stream* m_oldStderr;

   String m_scriptName;
   String m_scriptPath;

   Module* m_sessionModule;
};

}
}

#endif

/* end of modulewopi.h */
