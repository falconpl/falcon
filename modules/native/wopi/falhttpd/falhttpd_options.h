/*
   FALCON - The Falcon Programming Language.
   FILE: falhttpd_options.h

   Micro HTTPD server providing Falcon scripts on the web.
   Implementation of option file

   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Wed, 24 Feb 2010 20:10:45 +0100
s
   -------------------------------------------------------------------
   (C) Copyright 2010: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef FALCON_HTTPD_OPTIONS_H_
#define FALCON_HTTPD_OPTIONS_H_

#ifdef _WIN32
#include <winsock2.h>
#include <windows.h>
#endif

#include <falcon/types.h>
#include <falcon/string.h>
#include <confparser_mod.h>

#include <list>

namespace Falcon {

class FalhttpdClient;
class FalhttpdRequestHandler;

class FalhttpOptions
{
public:

   FalhttpOptions();
   ~FalhttpOptions();

   bool init( int argc, char* argv[] );
   void setIndexFile( const Falcon::String& fName );
   //bool loadConfig( const String& cfgFile );

   void parseIni();
   void parseMimeTypes();
   void parseRedirects();

   /** Change the file to be loaded.
   */
   bool remap( Falcon::String& sFname ) const;

   /** Determines the mime type of a file being loaded.
   */
   bool findMimeType( const Falcon::String& fname, Falcon::String& mtype ) const;

   /** Gets a registered request handler for the given file. */
   FalhttpdRequestHandler* getHandler( const Falcon::String& sFile, FalhttpdClient* cli ) const;

   void addMimeType( const String& sKey, const String& sValue );
   void addRedirect( const String& sKey, const String& sValue );

   Falcon::int64 m_maxUpload;
   Falcon::int64 m_maxMemUpload;
   Falcon::String m_sUploadPath;
   Falcon::String m_loadPath;
   Falcon::String m_homedir;
   Falcon::String m_configFile;
   Falcon::String m_sErrorDesc;
   Falcon::String m_sLogFiles;
   Falcon::String m_sIface;
   Falcon::String m_sIndexFile;
   Falcon::String m_sTextEncoding;
   Falcon::String m_sSourceEncoding;
   Falcon::String m_sAppDataDir;

   std::list<Falcon::String> m_lIndexFiles;

   int m_nPort;
   int m_logLevel;
   int m_nTimeout;
   bool m_bQuiet;
   bool m_bHelp;
   bool m_bSysLog;

   bool m_bAllowDir;

private:
   bool checkBool( const Falcon::String& b );
   ConfigFile m_cfg;

   class MimeType
   {
   public:
      Falcon::String m_def;
      std::list<Falcon::String> m_lWildcards;

      MimeType( const Falcon::String &def, const Falcon::String& wcard );
      bool match( const Falcon::String &name ) const;
   };

   class Redirect
   {
   public:
      Falcon::String m_sPath;
      Falcon::String m_sScript;

      Redirect( const Falcon::String& p, const Falcon::String& s ):
         m_sPath( p ),
         m_sScript( s )
      {}
   };

   std::list<MimeType> m_lMimeTypes;
   std::list< Redirect > m_lRedirects;
};

}

#endif

/* end of falhttpd_options.h */
