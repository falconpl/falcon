/*
   FALCON - The Falcon Programming Language.
   FILE: cgi_options.h

   Falcon CGI program driver - Options for the CGI module.

   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 21 Feb 2010 13:22:44 +0100

   -------------------------------------------------------------------
   (C) Copyright 2010: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/


#ifndef CGI_OPTIONS_H_
#define CGI_OPTIONS_H_

#include <falcon/string.h>
#include <falcon/wopi/session_manager.h>

class CGIOptions
{
public:
   CGIOptions();
   virtual ~CGIOptions();

   bool init( int argc, char* argv[] );

   Falcon::int64 m_maxUpload;
   Falcon::int64 m_maxMemUpload;
   Falcon::String m_sUploadPath;
   Falcon::String m_sScritpName;
   Falcon::String m_sMainScript;

   Falcon::WOPI::SessionManager* m_smgr;
};


#endif /* CGI_OPTIONS_H_ */

/* end of cgi_options.h */
