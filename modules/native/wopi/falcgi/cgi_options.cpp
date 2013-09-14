/*
   FALCON - The Falcon Programming Language.
   FILE: cgi_options.cpp

   Falcon CGI program driver - Options for the CGI module.

   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 21 Feb 2010 13:22:44 +0100

   -------------------------------------------------------------------
   (C) Copyright 2010: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/


#include "cgi_options.h"
#include <falcon/setup.h>
#include <falcon/path.h>

CGIOptions::CGIOptions():
   m_smgr( 0 )
{
   // provide some defaults
   m_maxUpload = 20000000;
   m_maxMemUpload = 5000;
#ifdef FALCON_SYSTEM_WIN
   m_sUploadPath = "/C:/TEMP";
#else
   m_sUploadPath = "/tmp";
#endif
}

CGIOptions::~CGIOptions()
{
}

bool CGIOptions::init( int argc, char* argv[] )
{
   if( argc == 0 )
      return false;

   m_sScritpName.bufferize( argv[0] );

   Falcon::Path ps( m_sScritpName );
   m_sMainScript = ps.file();

   return true;
}


/* end of cgi_options.cpp */
