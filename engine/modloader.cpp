/*
   FALCON - The Falcon Programming Language.
   FILE: modloader.cpp

   Module loader and reference resolutor.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Mon, 01 Aug 2011 11:45:31 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#undef SRC
#define SRC "engine/modloader.cpp"

#include <falcon/modloader.h>
#include <falcon/modspace.h>
#include <falcon/modcompiler.h>
#include <falcon/famloader.h>
#include <falcon/dynloader.h>
#include <falcon/sys.h>
#include <falcon/engine.h>
#include <falcon/transcoder.h>
#include <falcon/ioerror.h>
#include <falcon/vfsiface.h>
#include <falcon/textreader.h>
#include <falcon/datareader.h>
#include <falcon/stream.h>
#include <falcon/genericerror.h>

#include <falcon/trace.h>
#include <falcon/fassert.h>

#include <deque>

namespace Falcon
{

class ModLoader::Private 
{
public: 
   Private() {}
   ~Private() {}
   
   typedef std::deque<String> PathList;
   PathList m_plist;
   
};


ModLoader::ModLoader( ModCompiler* mc, FAMLoader* faml, DynLoader* dld ):
   _p( new Private )
{
   init(".", mc, faml, dld );
}


ModLoader::ModLoader( const String &path, ModCompiler* mc, FAMLoader* faml, DynLoader* dld ):
   _p( new Private )
{
   init( path, mc, faml, dld );
}


ModLoader::~ModLoader()
{
   delete _p;
}


void ModLoader::init ( const String &path, ModCompiler* mc, FAMLoader* faml, DynLoader* dld )
{
   static Engine* engine = Engine::instance();
   
   setSearchPath(path);
   if( mc == 0 ) mc = new ModCompiler;
   if( faml == 0 ) faml = new FAMLoader;
   if( dld == 0 ) dld = new DynLoader;
 
   m_compiler = mc;
   m_famLoader = faml; 
   m_dynLoader = dld;
   
   m_encName = "C";
   m_tcoder = engine->getTranscoder( m_encName );
   fassert( m_tcoder != 0 );
}


Module* ModLoader::loadName( const String& name, t_modtype type )
{
   String modName = name;
   
   // change "." into "/"
   length_t pos1 = modName.find( '.' );
   while( pos1 != String::npos )
   {
      modName.setCharAt( pos1, '/' );
      pos1 = modName.find( '.', pos1+1 );      
   }
   
   return loadFile( modName, type, true );
}


Module* ModLoader::loadFile( const String& path, t_modtype type, bool bScan )
{
   URI uri(path);
   return loadFile( uri, type, bScan );
}
   

Module* ModLoader::loadFile( const URI& uri, t_modtype type, bool bScan )
{
   URI tgtUri;
   
   // is the file absolute?
   Path path( URI::URLDecode( uri.path() ) );
   if( path.isAbsolute() || ! bScan )
   {      
      t_modtype etype = checkFile_internal( uri, type, tgtUri );
      if( etype != e_mt_none )
      {
         return load_internal( "", tgtUri, etype );
      }
   }
   else
   {
      // Search the file in the path elements.
      Private::PathList& plist = _p->m_plist;
      Private::PathList::iterator iter = plist.begin();
      while( iter != plist.end() )
      {
         URI location( *iter + "/" + uri.path() );  
         
         
         if( location.isValid() )
         {
            TRACE( "Scanning for module %s with type %d ", location.encode().c_ize(), type );
        
            t_modtype etype = checkFile_internal( location, type, tgtUri );
            if( etype != e_mt_none )
            {
               return load_internal( *iter, tgtUri, etype );
            }
         }
         else
         {
            TRACE( "URI not valid: %s", location.encode().c_ize() );
         }
         ++iter;
      }            
   }
   
   // We didn't find anything to be loaded.
   return 0;
}


void ModLoader::setSearchPath( const String &path )
{
   Private::PathList& plist = _p->m_plist;
   
   plist.clear();
   addSearchPath( path );
}

void ModLoader::addFalconPath()
{
   String envpath;
   bool hasEnvPath = Sys::_getEnv( "FALCON_LOAD_PATH", envpath );

   if ( hasEnvPath )
   {
      addSearchPath( envpath );
   }
   else {
      addSearchPath( FALCON_DEFAULT_LOAD_PATH );
   }
}

void ModLoader::addSearchPath( const String &path )
{
   Private::PathList& plist = _p->m_plist;
   
   // clear the path
   m_path = "";
   
   length_t pos0 = 0;
   length_t pos = path.find( ';' );
   while( pos != String::npos )
   {
      plist.push_back(path.subString( pos0, pos ) );
      pos0 = pos+1;
      pos = path.find( ';', pos0 );
   }

   // Push the last one
   if ( pos0 < path.length() )
   {
      plist.push_back(path.subString( pos0 ) );
   }
}


void ModLoader::addDirectoryFront( const String &directory )
{
   m_path = "";
   Private::PathList& plist = _p->m_plist;
   plist.push_front( directory );
}


void ModLoader::addDirectoryBack( const String &directory )
{
   m_path = "";
   Private::PathList& plist = _p->m_plist;
   plist.push_back( directory );
}
   

const String& ModLoader::getSearchPath() const
{
   if( m_path == "" )
   {
      const Private::PathList& plist = _p->m_plist;
      Private::PathList::const_iterator iter = plist.begin();
      while( iter != plist.end() )
      {
         if( m_path.size() != 0 )
         {
            m_path += ";";
         }
         
         m_path += *iter;
         ++iter;
      }
   }
   
   return m_path;
}
   


void ModLoader::pathToName( const String &path, const String &modFile, String &modName )
{
   // Chop away the topmost part of the path.
   if( modFile.find( path ) == 0 )
   {
      modName = modFile.subString( path.length()+1 );
   }
   else
   {
      modName = modFile;
   }
   
   // chop away ./ or /
   if( modName.find( "./" ) == 0 )
   {
      modName = modName.subString(2);
   }
   else if( modName.find( '/' ) == 0 )
   {
      modName = modName.subString(1);
   }
   
   // chop away terminal extension.
   length_t pos1 = modName.rfind( '.' );
   length_t pos2 = modName.rfind( '/' );
   if ( pos1 != String::npos && 
         (pos2 == String::npos || pos2 < pos1 ) )
   {
      modName.subString(0, pos1);
   }
   
   // change "/" into .
   pos1 = modName.find( '/' );
   while( pos1 != String::npos )
   {
      modName.setCharAt( pos1, '.' );
      pos1 = modName.find( '/', pos1+1 );      
   }
}


ModLoader::t_modtype ModLoader::checkFile_internal( 
      const URI& uri, ModLoader::t_modtype type, URI& foundUri )
{
   static VFSIface* vfs = &Engine::instance()->vsf();
   
   // if we have a type, just check if the beast exists.
   if( type != ModLoader::e_mt_none )
   {
      if( vfs->fileType( uri, true ) == FileStat::_normal )
      {
         foundUri = uri;
         return type;
      }
      
      return e_mt_none;
   }
   
   // else, try to find the required file, in priority order.
   FileStat stats[4];
   URI uris[4];
   t_modtype types[] = { e_mt_source, e_mt_vmmod, e_mt_binmod, e_mt_ftd };
   Path path( URI::URLDecode(uri.path()) );
   
   uris[0] = uri; path.ext( "fal" ); uris[0].path( path.encode() );
   uris[1] = uri; path.ext( m_famExt ); uris[1].path( path.encode() );
   uris[3] = uri; path.ext( m_ftdExt ); uris[3].path( path.encode() );
   // here we modify the filename, it must be done for last.
   uris[2] = uri; path.fileext( path.file() + "_fm." + DynLoader::sysExtension() ); 
                    uris[2].path( path.encode() );
   
   // the files we should look at depends on our working mode.
   switch( m_useSources )
   {
      case e_us_newer:
         vfs->readStats( uris[0], stats[0], true );
         vfs->readStats( uris[1], stats[1], true );
         vfs->readStats( uris[2], stats[2], true );
         vfs->readStats( uris[3], stats[3], true );
         break;
         
      case e_us_always:
         vfs->readStats( uris[0], stats[0], true );
         vfs->readStats( uris[3], stats[3], true );
         break;
         
      case e_us_never:
         vfs->readStats( uris[1], stats[1], true );
         vfs->readStats( uris[2], stats[2], true );
         break;
   }

   // who is the winner?
   int bestUri = -1;
   TimeStamp best;
   for( int i = 0; i < 4; ++i )
   {
      FileStat& st = stats[i];
      // was this stat found? 
      if( st.type() != FileStat::_notFound )
      {
         // if yes, check if we should use it.
         if( bestUri == -1 || stats[bestUri].mtime().compare(st.mtime()) < 0 )
         {
            bestUri = i;
         }
      }
   }
   
   // Not found? -- ignore.
   if( bestUri == -1 )
   {
      return e_mt_none;
   }

   // Found? -- get the found uri and return the type.
   foundUri = uris[bestUri];
   return types[ bestUri ];
}


Module* ModLoader::load_internal( 
      const String& prefixPath, const URI& uri, ModLoader::t_modtype type )
{
   static VFSIface* vfs = &Engine::instance()->vsf();
  
   String modName;
   // The module name depends on the prefix path.
   // if the scheme is not in the prefix, then we should just use the path.
   if( prefixPath.find( uri.scheme() ) == 0 )
   {
      pathToName( prefixPath, uri.encode(), modName );
   }
   else
   {
      pathToName( prefixPath, uri.path(), modName );
   }
   
   // Use the right device depending on the file type.
   switch( type )
   {
      case e_mt_source:
      case e_mt_ftd:
      {
         // TODO: Treat FTD
         Stream* ins = vfs->openRO( uri );
         if( ins == 0 )
         {
            throw makeError( e_nofile, __LINE__, uri.encode() );
         }
         ins->shouldThrow(true);
         TextReader* input = new TextReader( ins, m_tcoder, true );
         // compiler gets the ownership of input.
         Module* output = m_compiler->compile( input, uri.encode(), modName );
         
         // for now, we just throw
         if( output == 0 )
         {
            throw m_compiler->makeError();
         }
         return output;
      }
         
      case e_mt_vmmod:
      {
         Stream* ins = vfs->openRO( uri );
         if( ins == 0 )
         {
            throw makeError( e_nofile, __LINE__, uri.encode() );
         }
         
         ins->shouldThrow(true);
         DataReader dr( ins, DataReader::e_LE, true );
         return m_famLoader->load( &dr, uri.encode(), modName );
      }
         
      case e_mt_binmod:         
         return m_dynLoader->load( uri.encode(), modName );
         
      default:
         fassert2(false, "Should not be here...");
         return 0;
   }
}
   

Error* ModLoader::makeError( int code, int line, const String &expl, int fsError )
{
   return new IOError( ErrorParam( code, line, SRC )
            .extra( expl )
            .origin( ErrorParam::e_orig_loader )
            .sysError( fsError )
         );
}

}

/* end of modloader.cpp */
