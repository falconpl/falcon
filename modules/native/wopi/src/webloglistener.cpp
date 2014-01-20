/*
   FALCON - The Falcon Programming Language.
   FILE: webloglistener.cpp

   Web Log Listener for WOPI.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 19 Jan 2014 17:59:47 +0100

   -------------------------------------------------------------------
   (C) Copyright 2014: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#include <falcon/wopi/webloglistener.h>
#include <list>

namespace Falcon {
namespace WOPI {

class WebLogListener::Private
{
public:
   typedef std::list<String> StringList;
   StringList m_list;

   Private() {}
   ~Private() {}
};

WebLogListener::WebLogListener()
{
   _p = new Private;
}

WebLogListener::~WebLogListener()
{
   delete _p;
}


void WebLogListener::renderLogs( TextWriter* target )
{
   if( hasLogs() )
   {
      target->writeLine("<hr/><h1>Log</h1><pre>\n");
      Private::StringList::iterator iter = _p->m_list.begin();
      while( iter != _p->m_list.end() )
      {
         String line;
         (*iter).replace("<", "&lt;", line );
         target->writeLine(line);
         ++iter;
      }
      target->writeLine("\n<pre><hr/>");
   }
}


bool WebLogListener::hasLogs() const
{
   return ! _p->m_list.empty();
}


void WebLogListener::onMessage( int fac, int lvl, const String& message )
{
   String target;
   Log::formatLog(fac,lvl,message,target);
   _p->m_list.push_back(target);
}

}
}

/* end of webloglistener.cpp */
