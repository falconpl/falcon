/*
   FALCON - The Falcon Programming Language.
   FILE: falhttpd_options.cpp

   Micro HTTPD server providing Falcon scripts on the web.
   Implementation of option file

   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Wed, 24 Feb 2010 20:10:45 +0100

   -------------------------------------------------------------------
   (C) Copyright 2010: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#include "falhttpd.h"
#include "falhttpd_options.h"
#include <falcon/engine.h>
#include <falcon/sys.h>
#include <falcon/wopi/mem_sm.h>
#include <falcon/wopi/utils.h>

#include "falhttpd_rh.h"
#include "falhttpd_filehandler.h"
#include "falhttpd_scripthandler.h"
#include "falhttpd_dirhandler.h"

#include <list>

namespace Falcon {

//TODO: set from .h
FalhttpOptions::FalhttpOptions():
   m_loadPath( "." ),
   m_sIface( "0.0.0.0" ),
   m_nPort(80),
   m_logLevel( 3 ),
   m_bQuiet( false ),
   m_bHelp( false ),
   m_bSysLog( true ),
   m_bAllowDir( true )
{
   m_sTextEncoding = "utf8";
   m_sSourceEncoding = "utf8";
   fassert( Engine::instance()->getTranscoder(m_sTextEncoding) != 0 );
   
   setIndexFile( "index.ftd;index.fal;index.html;index.htm" );
}

FalhttpOptions::~FalhttpOptions()
{
}


bool FalhttpOptions::init( int argc, char* argv[] )
{
   Falcon::String res;
   Falcon::String logLevel;
   Falcon::String sPort, sTimeout;

   // first thing, pre-configure falcon load path.
   if ( Falcon::Sys::_getEnv("FALCON_LOAD_PATH", res ) )
      m_loadPath = res;


   // then read the parameters
   Falcon::String* pParam = 0;
   for( int i = 1; i < argc; ++i )
   {
      char* param = argv[i];

      if ( pParam != 0 )
      {
         pParam->bufferize(param);
         pParam = 0;
         continue;
      }

      if( param[0] == '-' )
      {
         switch( param[1] )
         {
         case 'A': pParam = &m_sAppDataDir; break;
         case 'C': pParam = &m_configFile; break;
         case 'D': pParam = &m_sLogFiles; break;
         case 'h': pParam = &m_homedir; break;
         case 'l': pParam = &logLevel; break;
         case 'i': pParam = &m_sIface; break;
         case 'L': pParam = &m_loadPath; break;
         case 'p': pParam = &sPort; break;
         case 'q': m_bQuiet = true; break;
         case 'S': m_bSysLog = false; break;
         case 't': pParam = &sTimeout; break;
         case 'e': pParam = &m_sTextEncoding; break;
         case 'E': pParam = &m_sSourceEncoding; break;
         case '?': m_bHelp = true; break;
         default:
            m_sErrorDesc = "Invalid command ";
            m_sErrorDesc += param;
            return false;
         }
      }
      else
      {
         if ( m_configFile.size() )
         {
            m_sErrorDesc = "Give just one config file";
            return false;
         }
         m_configFile.bufferize( param );
      }
   }

   if( pParam != 0 )
   {
      m_sErrorDesc = "Missing mandatory parameter for command ";
      m_sErrorDesc += *pParam;
      return false;
   }

   // log level post processing
   Falcon::int64 ll;
   if( logLevel.size() != 0 )
   {
      if( logLevel.size() != 0 && logLevel.parseInt( ll ) )
         m_logLevel = (int) ll;
   }

   // port address post processing
   if( sPort.size() != 0 )
   {
      if( sPort.parseInt( ll ) )
      {
         m_nPort = (int) ll;
      }
      else
      {
         m_sErrorDesc = "Invalid value for -p " + sPort;
         return false;
      }
   }

   // port address post processing
   if( sTimeout.size() != 0 )
   {
      if( sTimeout.parseInt( ll ) )
      {
         m_nTimeout = (int) ll;
      }
      else
      {
         m_sErrorDesc = "Invalid value for -t " + sTimeout;
         return false;
      }
   }

   if( ! m_configFile.empty() )
   {
      m_wopiIni = m_configFile;
      parseIni();
      parseMimeTypes();
   }


   if ( ! m_wopiIni.empty() )
   {
      parseWopiIni();
   }

   // add some sensible MIME default.
   m_lMimeTypes.push_back( MimeType( "text/html", "*.html;*.htm" ) );
   m_lMimeTypes.push_back( MimeType( "text/css", "*.css" ) );
   m_lMimeTypes.push_back( MimeType( "text/javascript", "*.js" ) );
   m_lMimeTypes.push_back( MimeType( "image/png", "*.png" ) );
   m_lMimeTypes.push_back( MimeType( "image/gif", "*.gif" ) );
   m_lMimeTypes.push_back( MimeType( "image/jpg", "*.jpg;*.jpeg" ) );
   m_lMimeTypes.push_back( MimeType( "image/tiff", "*.tif;*.tiff" ) );
   m_lMimeTypes.push_back( MimeType( "text/plain", "*" ) );

   if ( Engine::instance()->getTranscoder(m_sTextEncoding) == 0 )
   {
      m_sErrorDesc += "Invalid text stream encoding " + m_sTextEncoding +"\n";
   }

   if ( Engine::instance()->getTranscoder(m_sSourceEncoding) == 0 )
   {
      m_sErrorDesc += "Invalid source script encoding " + m_sSourceEncoding +"\n";
   }

   return m_sErrorDesc.empty();
}


void FalhttpOptions::setIndexFile( const Falcon::String& fName )
{
   m_lIndexFiles.clear();
   
   Falcon::uint32 pos1 = fName.find( ";" );
   Falcon::uint32 pos = 0;

   while( pos1 != Falcon::String::npos )
   {
      m_lIndexFiles.push_back( fName.subString( pos, pos1 ) );
      pos = pos1 + 1;
      pos1 = fName.find( ";", pos );
   }

   m_lIndexFiles.push_back( fName.subString( pos ) );
   m_sIndexFile = fName;

}


bool FalhttpOptions::remap( Falcon::String& sFname ) const
{
   // first, let's handle redirection as-is.
   std::list<Redirect>::const_iterator ired = m_lRedirects.begin();
   while( ired != m_lRedirects.end() )
   {
      if( sFname.startsWith(ired->m_sPath) )
      {
         // we found it.
         sFname = ired->m_sScript;
         return true;
      }

      ++ired;
   }

   // find the file.
   Falcon::Path path( sFname );
   if( sFname == "" ) {
      path.fulloc("/");
   }

   path.fulloc( m_homedir + path.fulloc() );

   sFname = path.encode();
   Falcon::FileStat stats;
   if( ! Engine::instance()->vfs().readStats( sFname, stats, true ) )
   {
      return false;
   }
   else
   {
      // Is the file a directory?
      if( stats.type() == Falcon::FileStat::_dir )
      {
         path.fulloc( sFname );
         // Do we have index files to try?
         std::list<Falcon::String>::const_iterator fni = m_lIndexFiles.begin();
         while( fni != m_lIndexFiles.end() )
         {
            path.file( *fni );
            if( Engine::instance()->vfs().readStats( path.encode(), stats, true ) )
            {
               break;
            }
            ++fni;
         }

         // return it as a directory?
         if( fni == m_lIndexFiles.end() )
         {
            sFname = path.fulloc();
            if( ! sFname.endsWith("/") )
            {
               sFname += "/";
            }
         }
         else
         {
            sFname = path.encode();
         }

      }
   }

   return true;
}


void FalhttpOptions::parseIni()
{
   try {

      Stream* stream = Engine::instance()->vfs().openRO( m_configFile );
      TextReader input(stream, Engine::instance()->getTranscoder(m_sSourceEncoding) );
      stream->decref();
      m_cfg.load(&input);
      if( m_cfg.mainSection() != 0 )
      {
         ConfigSection* cs = m_cfg.mainSection();
         cs->getValue( "HomeDir", m_homedir );
         cs->getValue( "Interface", m_sIface );
         cs->getValue( "PersistentDataDir", m_sAppDataDir );
         cs->getValue( "WopiINI", m_wopiIni );
      }
   }
   catch( Error* err )
   {
      m_sErrorDesc += "Error reading configuration file: " + err->describe();
   }
}


void FalhttpOptions::parseWopiIni()
{
   try {
      Stream* stream = Engine::instance()->vfs().openRO( m_wopiIni );
      TextReader input(stream, Engine::instance()->getTranscoder(m_sSourceEncoding) );
      stream->decref();
      String errors;
      if( ! m_templateWopi.configFromIni( &input, errors ) )
      {
         m_sErrorDesc += "Error reading WOPI configuration: " + errors;
      }
   }
   catch( Error* err )
   {
      m_sErrorDesc += "Error reading configuration file: " + err->describe();
   }
}


void FalhttpOptions::parseMimeTypes()
{

   ConfigSection* cs = m_cfg.getSection("MIME");
   if( cs != 0 )
   {
      class KN: public ConfigSection::KeyEnumerator
      {
      public:
         KN( FalhttpOptions* opts ):
            m_opts(opts)
         {}

         virtual ~KN(){}

         void operator() (const String& key, const String& value)
         {
            m_opts->addMimeType(key, value);
         }

         FalhttpOptions* m_opts;
      }
      rator( this );

      cs->enumerateKeys(rator);
   }
}



void FalhttpOptions::parseRedirects()
{
   Falcon::String sValue;
   ConfigSection* cs = m_cfg.getSection("REDIRECT");

   class KN: public ConfigSection::KeyEnumerator
   {
   public:
      KN( FalhttpOptions* opts ):
         m_opts(opts)
      {}

      virtual ~KN(){}

      void operator() (const String& key, const String& value)
      {
         m_opts->addRedirect(key, value);
      }

      FalhttpOptions* m_opts;
   }
   rator( this );

   cs->enumerateKeys(rator);
}

void FalhttpOptions::addMimeType( const Falcon::String& sKey, const Falcon::String& sValue )
{
   LOGI( "Parsing mime type " + sKey + " = " + sValue );

   if ( sValue == "*" )
   {
      m_lMimeTypes.push_back( MimeType( sKey, sValue ) );
   }
   else
   {
      m_lMimeTypes.push_front( MimeType( sKey, sValue ) );
   }
}


void FalhttpOptions::addRedirect( const Falcon::String& sKey, const Falcon::String& sValue )
{
   Falcon::String sPath = sKey;

   // remove the * so that we get the whole path
   if( sPath.endsWith("*") )
   {
      sPath.remove(sPath.length()-1,1);
      m_lRedirects.push_back( Redirect( sPath, sValue ) );
   }
   else
   {
      sPath += "/";
      m_lRedirects.push_front( Redirect( sPath, sValue ) );
   }

   LOGI( "Adding redirect script handler "+ sValue + " for queries in "  + sPath );

}

FalhttpOptions::MimeType::MimeType( const Falcon::String& sType, const Falcon::String& sValue ):
      m_def(sType)
{
   Falcon::uint32 pos = 0, pos1;

   do
   {
      pos1 = sValue.find( ";", pos );
      Falcon::String sPart = sValue.subString( pos, pos1 );
      sPart.trim();
      m_lWildcards.push_back(sPart);
      pos = pos1+1;
   }
   while( pos1 != Falcon::String::npos );

}


bool FalhttpOptions::MimeType::match( const Falcon::String& sFname ) const
{
   std::list<Falcon::String>::const_iterator pos = m_lWildcards.begin();
   while( pos != m_lWildcards.end() )
   {
      if( sFname.wildcardMatch( *pos ) )
         return true;

      ++pos;
   }

   return false;
}

bool FalhttpOptions::findMimeType( const Falcon::String& sFname, Falcon::String& type ) const
{
   std::list<MimeType>::const_iterator melem = m_lMimeTypes.begin();
   while( melem != m_lMimeTypes.end() )
   {
      const MimeType& mtype = *melem;
      if( mtype.match( sFname ) )
      {
         type = mtype.m_def;
         return true;
      }
      ++melem;
   }

   return false;
}


bool FalhttpOptions::checkBool( const Falcon::String& s )
{
   Falcon::String upper = s;
   upper.upper();

   return (upper == "T") || (upper == "TRUE") || (upper == "1") || (upper == "ON");
}


FalhttpdRequestHandler* FalhttpOptions::getHandler(
      const Falcon::String& sFile, FalhttpdClient* cli ) const
{
   if ( sFile.endsWith("/") && m_bAllowDir )
   {
      return new DirHandler( sFile, cli );
   }
   else if( sFile.endsWith(".fal") || sFile.endsWith(".fam") || sFile.endsWith(".ftd") )
   {
      return new ScriptHandler( sFile, cli );
   }
      
   return new FileHandler( sFile, cli );
}

}

/* end of falhttpd_options.cpp */
