/*
   FALCON - The Falcon Programming Language.
   FILE: flc_modloader.cpp

   Module loader
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: 2004-08-20

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#include <falcon/setup.h>
#include <falcon/modloader.h>
#include <falcon/error.h>
#include <falcon/fstream.h>
#include <falcon/module.h>
#include <falcon/dll.h>
#include <falcon/fstream.h>
#include <falcon/sys.h>
#include <falcon/pcodes.h>
#include <falcon/timestamp.h>
#include <falcon/transcoding.h>
#include <falcon/stringstream.h>
#include <falcon/gencode.h>
#include <falcon/uri.h>
#include <falcon/vfsprovider.h>
#include <falcon/streambuffer.h>
#include <falcon/stdstreams.h>

#include <memory>



#define BINMODULE_EXT "_fm"

namespace Falcon
{

ModuleLoader::ModuleLoader():
   m_alwaysRecomp( false ),
   m_compMemory( true ),
   m_saveModule( true ),
   m_saveMandatory( false ),
   m_detectTemplate( true ),
   m_forceTemplate( false ),
   m_delayRaise( false ),
   m_ignoreSources( false ),
   m_saveRemote( false ),
   m_compileErrors(0)
{
   m_path.deletor( string_deletor );
   setSearchPath( "." );
}


ModuleLoader::ModuleLoader( const String &path ):
   m_alwaysRecomp( false ),
   m_compMemory( true ),
   m_saveModule( true ),
   m_saveMandatory( false ),
   m_detectTemplate( true ),
   m_forceTemplate( false ),
   m_delayRaise( false ),
   m_ignoreSources( false ),
   m_saveRemote( false ),
   m_compileErrors(0)
{
   m_path.deletor( string_deletor );
   setSearchPath( path );
}

ModuleLoader::ModuleLoader( const ModuleLoader &other ):
   m_alwaysRecomp( other.m_alwaysRecomp ),
   m_compMemory( other.m_compMemory ),
   m_saveModule( other.m_saveModule ),
   m_saveMandatory( other.m_saveMandatory ),
   m_detectTemplate( other.m_detectTemplate ),
   m_forceTemplate( other.m_forceTemplate ),
   m_delayRaise( other.m_delayRaise ),
   m_ignoreSources( other.m_ignoreSources ),
   m_saveRemote( other.m_saveRemote ),
   m_compileErrors( other.m_compileErrors )
{
   setSearchPath( other.getSearchPath() );
}


ModuleLoader::~ModuleLoader()
{
}


ModuleLoader *ModuleLoader::clone() const
{
   return new ModuleLoader( *this );
}


void ModuleLoader::getModuleName( const String &path, String &modName )
{
   // .../modname.xxx
   // we need at least a char.
   if ( path.length() == 0 )
   {
      modName = "";
      return;
   }

   int32 dotpos = path.rfind( "." );
   int32 slashpos = path.rfind( "/" );
   if ( dotpos == -1 )
      dotpos = path.length();
   // legal also if slashpos < 0

   modName = path.subString( slashpos + 1, dotpos );
}


void ModuleLoader::addFalconPath()
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

void ModuleLoader::setSearchPath( const String &path )
{
   m_path.clear();
   addSearchPath( path );
}

void ModuleLoader::addSearchPath( const String &path )
{
   // subdivide the path by ';'
   int32 pos = 0, pos1 = 0;

   // nothing to add ?
   if ( path == "" )
      return;;

   while( true )
   {
      String *tmp;

      pos1 = path.find( ";", pos );
      if ( pos1 ==  -1 )
      {
         tmp = new String( path, pos );
         tmp->bufferize();
         m_path.pushBack( tmp );
         break;
      }

      if ( pos1 == -1 ) {
         tmp = new String( path, pos );
         tmp->bufferize();
         m_path.pushBack( tmp );
         break;
      }

      if ( pos1 > pos ) {
         tmp = new String( path, pos, pos1 );
         tmp->bufferize();
         m_path.pushBack( tmp );
      }
      pos = pos1+1;
   }
}

void ModuleLoader::getSearchPath( String &tgt ) const
{
   tgt.size(0);

   ListElement *path_elem = m_path.begin();
   while ( path_elem != 0 )
   {
      String *pathp = (String *) path_elem->data();
      tgt += *pathp;
      path_elem = path_elem->next();
      if ( path_elem != 0 )
         tgt += ";";
   }
}


Stream *ModuleLoader::openResource( const String &path, t_filetype type )
{
   // Get the uri
   URI furi( path );

   if ( !furi.isValid() )
   {
      raiseError( e_malformed_uri, path ); // upstream will fill the module
   }

   // find the appropriage provider.
   VFSProvider* vfs = Engine::getVFS( furi.scheme() );
   if ( vfs == 0 )
   {
      raiseError( e_unknown_vfs, path );  // upstream will fill the module
   }

   Stream *in = vfs->open( furi, VFSProvider::OParams().rdOnly() );

   if ( in == 0 )
   {
      throw vfs->getLastError();
   }

   if ( type == t_source || type == t_ftd )
   {
      if ( m_srcEncoding != "" )
      {
         // set input encoding
         Stream *inputStream = TranscoderFactory( m_srcEncoding, in, true );
         if( inputStream != 0 )
            return AddSystemEOL( inputStream );

         delete in;
         raiseError( e_unknown_encoding, m_srcEncoding ); // upstream will fill the module
      }
      else
         return AddSystemEOL( in );
   }

   return in;
}


ModuleLoader::t_filetype ModuleLoader::fileType( const String &fext )
{
   String ext = fext;
   ext.lower();

   if ( ext == "fal" )
   {
      return t_source;
   }
   else if ( ext == "ftd" )
   {
      return t_ftd;
   }
   else if ( ext == DllLoader::dllExt() )
   {
      return t_binmod;
   }
   else if ( ext == "fam" )
   {
      return t_vmmod;
   }

   return t_none;
}


Module *ModuleLoader::loadName( const String &module_name, const String &parent_name )
{
   String file_path;
   String nmodName;

   // prevent doing a crashy thing.
   if ( module_name.length() == 0 )
      throw new CodeError( ErrorParam( e_modname_inv ).extra( module_name ).origin( e_orig_loader ).
            module( "(loader)" ) );

   nmodName = Module::absoluteName( module_name, parent_name );

   String path_name;
   String expName = nmodName;

   // expand "." names into "/"
   uint32 pos = expName.find( "." );
   while( pos != String::npos )
   {
      expName.setCharAt( pos, '/' );
      pos = expName.find( ".", pos + 1 );
   }


   Module *mod = loadFile( expName, t_none, true );
   mod->name( nmodName );

   return mod;
}


bool ModuleLoader::scanForFile( URI &origUri, VFSProvider* vfs, t_filetype &type, FileStat &fs )
{
   // loop over the possible extensions and pick the newest.
   const char *exts[] = { "ftd", "fal", "fam", DllLoader::dllExt(), 0 };
   const t_filetype ftypes[] = { t_ftd, t_source, t_vmmod, t_binmod, t_none };

   TimeStamp tsNewest;
   const char **ext = exts;
   const t_filetype *ptf = ftypes;

   // skip source exensions if so required.
   if ( ignoreSources() )
   {
      ext++; ext++;
      ptf++; ptf++;
   }

   while( *ext != 0 )
   {
      origUri.pathElement().setExtension( *ext );

      // did we find it?
      if ( *ptf == t_binmod )
      {
         // for binary module, add the extra module identifier.
         URI copy(origUri);
         copy.pathElement().setFile( copy.pathElement().getFile() + BINMODULE_EXT );
         if( vfs->readStats( copy, fs ) )
         {
            origUri.pathElement().setFile( copy.pathElement().getFile() );
            type = t_binmod;
            return true;
         }
      }
      else if ( vfs->readStats( origUri, fs ) )
      {
         type = *ptf;
         return true;
      }

      // get next extension and file type
      ext++;
      ptf++;
   }

   return false;
}

Module *ModuleLoader::loadFile( const URI& uri, t_filetype type, bool scan )
{
   URI origUri = uri;
   String file_path;
   t_filetype tf = t_none;

   VFSProvider *vfs = Engine::getVFS( origUri.scheme() );
   if ( vfs == 0 )
   {
      throw new CodeError( ErrorParam( e_unknown_vfs )
         .extra( uri.scheme() )
            .origin( e_orig_loader )
            );
   }

   // Check wether we have absolute files or files to be searched.

   FileStat foundStats;
   bool bFound = false;

   // If we don't have have an absolute path,
   if ( ! origUri.pathElement().isAbsolute() )
   {
      // ... and if scan is false, we must add our relative path to absolutize it.
      if ( ! scan )
      {
         String curdir;
         int32 error;
         if ( ! Sys::fal_getcwd( curdir, error ) )
         {
            throw new IoError( ErrorParam( e_io_error )
               .extra( Engine::getMessage( msg_io_curdir ) )
               .origin( e_orig_loader )
               .sysError( error )
               );
         }
         origUri.path( curdir + "/" + origUri.path() );
      }
   }
   else
   {
      // absolute path force scan to be off, just to be sure.
      scan = false;
   }

   // we are interested in knowing if a default extension was given,
   // in that case we won't go searching for that.
   bool bHadExtension = origUri.pathElement().getExtension() != "";

   // if we don't have to scan in the path...
   if ( ! scan )
   {
      // ... if type is t_none, we may anyhow scan for a proper extension.
      if( (type == t_none || type == t_defaultSource) && ! bHadExtension )
      {
         URI saveUri = origUri;
         bFound = scanForFile( origUri, vfs, tf, foundStats );
         if ( ! bFound && type == t_defaultSource )
         {
            origUri = saveUri;

            // scanforfile may
            bFound = vfs->readStats( origUri, foundStats );
            if ( bFound )
               tf = t_source;
         }
      }
      else {
         // just check if the file exists.
         tf = type == t_none ? fileType(origUri.pathElement().getExtension()) : type;
         bFound = vfs->readStats( origUri, foundStats );
      }
   }
   // shall we scan for a file?
   else
   {
      // ready to scan the list of directories;
      ListElement *path_elem = m_path.begin();
      String oldPath = origUri.pathElement().getLocation();
      while ( (! bFound) && path_elem != 0 )
      {
         String *pathp = (String *) path_elem->data();
         origUri.pathElement().setFullLocation( *pathp );
         origUri.pathElement().extendLocation( oldPath  );

         // If we originally had an extension, we must not add it.
         if ( bHadExtension )
         {
            // if the thing exists...
            if( ( bFound = vfs->readStats( origUri, foundStats ) ) )
            {
               // ... set the file type, either on our default or on the found extension.
               tf = (type == t_none || type == t_defaultSource ) ?
                     fileType( origUri.pathElement().getExtension() ) : type;
            }
         }
         else
         {
            // we must can the possible extensions in this directory
            bFound = scanForFile( origUri, vfs, tf, foundStats );
         }

         path_elem = path_elem->next();
      }
   }


   Module *mod = 0;
   // did we found a file?
   if( bFound )
   {
      // Ok, TF should be the file type.
      switch( tf )
      {
      case t_source: case t_ftd: case t_defaultSource:
         {
            FileStat fs;
            // should we load a .fam instead?
            URI famUri = origUri;
            famUri.pathElement().setExtension( "fam" );
            if ( ! alwaysRecomp()
                 && ( ignoreSources()
                      || (vfs->readStats( famUri, fs ) && *fs.m_mtime >= *foundStats.m_mtime) )
               )
            {
               try {
                  mod = loadModule( famUri.get() );
               }
               catch( Error *e )
               {
                  // well, our try wasn't cool. try with the source.
                  e->decref();
                  mod = loadSource( origUri.get() );
               }
            }
            else
            {
               mod = loadSource( origUri.get() );
            }
         }
         break;

      case t_vmmod: mod = loadModule( origUri.get() ); break;
      case t_binmod: mod = loadBinaryModule( URI::URLDecode( origUri.get() ) ); break;

      default:
         // we have not been able to find it
         // -- report the actual file that caused the problem.
         raiseError( e_unrec_file_type, origUri.get() );
      }
   }
   else
   {
      raiseError( e_nofile, URI::URLDecode(uri.get()) );
   }

   // in case of errors, we already raised
   String modName;
   getModuleName( origUri.get(), modName );
   mod->name( modName );
   mod->path( origUri.get() );

   // try to load the language table.
   if ( m_language != "" && mod->language() != m_language )
   {
      // the function may fail, but it won't raise.
      loadLanguageTable( mod, m_language );
   }

   // in case of errors, we already raised
   return mod;
}


Module *ModuleLoader::loadFile( const String &module_path, t_filetype type, bool scan )
{
   // Preliminary filtering -- get the URI and the filesystem.
   URI origUri;
   origUri.parse( module_path, true, false );
  
   if ( ! origUri.isValid() )
   {
      throw new CodeError( ErrorParam( e_malformed_uri )
            .extra( module_path )
            .origin( e_orig_loader )
            );
   }
   
   return loadFile( origUri, type, scan );
}



bool ModuleLoader::loadLanguageTable( Module *module, const String &language )
{
   String langFileName;

   // try to find the .ftr file
   uint32 posDot = module->path().rfind( "." );
   uint32 posSlash = module->path().rfind( "/" );
   if ( posDot == String::npos || ( posSlash != String::npos && posDot < posSlash ) )
   {
      langFileName = module->path() + ".ftr";
   }
   else {
      langFileName = module->path().subString(0, posDot );
      langFileName += ".ftr";
   }

   if( applyLangTable( module, langFileName ) )
   {
      module->language( language );
      return true;
   }

   return false;
}

inline int32 xendianity( bool sameEndianity, int32 val )
{
   return sameEndianity ? val :
      (val >> 24) |
      ((val & 0xFF0000) >> 8) |
      ((val & 0xFF00 ) << 8) |
      (val << 24);
}

//TODO: add some diags.
bool ModuleLoader::applyLangTable( Module *mod, const String &file_path )
{
   URI fsuri( file_path );
   fassert( fsuri.isValid() );
   VFSProvider* vfs = Engine::getVFS( fsuri.scheme() );
   fassert( vfs != 0 );

   // try to open the required file table.
   Stream *fsin_p = vfs->open( fsuri, VFSProvider::OParams().rdOnly() );
   if( fsin_p == 0 )
      return false;
      
   std::auto_ptr<Stream> fsin( fsin_p );

   // check if this is a regular tab file.
   char buf[16];
   buf[5] = 0;
   if ( fsin->read( buf, 5 ) != 5 || String( "TLTAB" ) != buf )
   {
      return false;
   }

   uint16 endianity;
   if( fsin->read( &endianity, 2 ) != 2 )
   {
      return false;
   }

   bool sameEndianity = endianity == 0xFBFC;

   // read the language table index.
   int32 sizeField;
   if( fsin->read( &sizeField, 4 ) != 4 )
      return false;

   int32 tableSize = xendianity( sameEndianity, sizeField );
   int32 tablePos = -1;
   for( int32 i = 0; i < tableSize; i++ )
   {
      // read language code and position in file
      if( fsin->read( buf, 5 ) != 5 ||
          fsin->read( &sizeField, 4 ) != 4 )
         return false;

      // is this our language code?
      if( m_language == buf )
      {
         tablePos = xendianity( sameEndianity, sizeField );
         break;
      }
   }

   // entry not found?
   if( tablePos < 0 )
      return false;

   uint32 headerSise = 5 + 2 + 4 + (tableSize * 9);
   uint32 filePos = headerSise + tablePos;
   fsin->seekBegin( filePos );

   // read the number of strings to be decoded.
   if( fsin->read( &sizeField, 4 ) != 4 )
      return false;

   int32 stringCount = xendianity( sameEndianity, sizeField );

   // read table and alter module.
   int32 allocated = 256;
   char *memBuf = (char *) memAlloc( allocated );

   // the most intelligent thing is that to modify the strings as they are in memory.
   // In this way, we don't have to alter already allocated string structures, and
   // we don't have to scan the map for the correct string entry.

   while ( stringCount > 0 )
   {
      // read ID
      if( fsin->read( &sizeField, 4 ) != 4 )
         break;
      int32 stringID = xendianity( sameEndianity, sizeField );
      if ( stringID < 0 || stringID >= mod->stringTable().size() )
         break;

      // read the string size
      if( fsin->read( &sizeField, 4 ) != 4 )
         break;
      int32 stringSize = xendianity( sameEndianity, sizeField );
      if ( stringSize < 0 )
         break;
      if ( stringSize == 0 )
         continue;

      // if the string size exeeds the allocated amount, fix it.
      if( stringSize >= allocated )
      {
         memFree( memBuf );
         allocated = stringSize + 1;
         memBuf = (char *) memAlloc( allocated );
      }

      // read the string
      if( fsin->read( memBuf, stringSize ) != stringSize )
         break;

      // zero the end so we have an utf8 string
      memBuf[ stringSize ] = 0;

      // finally, place it in the right place
      if ( ! mod->stringTable().getNonConst( stringID )->fromUTF8( memBuf ) )
         break;

      stringCount --;
   }

   memFree( memBuf );
   return stringCount == 0;
}



Module *ModuleLoader::loadModule( const String &path )
{
   // May throw on error.
   Module *mod = 0;

   {
      // loadModule may throw, so we need an autoptr not to leak in case of errors.
      std::auto_ptr<Stream> in( openResource( path, t_vmmod ));
      mod = loadModule( in.get() );
   }
   fassert( mod != 0 );

   String modName;
   getModuleName( path, modName );
   mod->name( modName );
   mod->path( path );

   if ( m_language != "" && mod->language() != m_language )
   {
      // This should not throw.
      loadLanguageTable( mod, m_language );
   }

   return mod;
}


Module *ModuleLoader::loadBinaryModule( const String &path )
{
   DllLoader dll;

   if ( ! dll.open( path ) )
   {
      String error;
      dll.getErrorDescription( error );
      error.prepend( path + ": " );
      raiseError( e_binload, error );
      return 0;
   }

   DllFunc dlfunc = dll.getSymbol( "falcon_module_init" );
   ext_mod_init func = (ext_mod_init) dlfunc.data();

   if ( func == 0 )
   {
      raiseError( e_binstartup, path );
      return 0;
   }

   Module *mod = func();

   if ( mod == 0 )
   {
      raiseError( e_bininit, path);
      return 0;
   }

   // Now I can pass the DLL instance to the module.
   mod->dllLoader().assign( dll );

   // and give the module its names.
   String modName;
   getModuleName( path, modName );
   mod->name( modName );
   mod->path( path );

   // as dll instance has been emptied, the DLL won't be closed till the
   // module lifetime comes to an end.
   return mod;
}


Module *ModuleLoader::loadModule( Stream *in )
{
   // try to open the file
   char c1, c2;

   in->read( &c1, 1 );
   in->read( &c2, 1 );

   if(c1 =='F' && c2 =='M')
   {
      Module *ret = loadModule_select_ver( in );
      return ret;
   }

   return 0;
}


Module *ModuleLoader::loadModule_select_ver( Stream *in )
{
   char c1, c2;

   in->read( &c1, 1 );
   in->read( &c2, 1 );

   // C1 and c2 now contain the version.
   // for now we can load only format PCODE

   if( c1 == FALCON_PCODE_VERSION && c2 == FALCON_PCODE_MINOR ) {
      return  loadModule_ver_1_0( in );
   }

   raiseError( e_modver, "" );
   return 0;
}

Module *ModuleLoader::loadModule_ver_1_0( Stream *in )
{
   Module *mod = new Module();
   if ( ! mod->load( in, true ) )
   {
      raiseError( e_modformat, "" );
   }
   return mod;
}


void  ModuleLoader::raiseError( int code, const String &expl, int fsError )
{
   throw new IoError( ErrorParam( code )
            .extra( expl )
            .origin( e_orig_loader )
            .sysError( fsError )
         );
}

Module *ModuleLoader::loadSource( const String &file )
{
   // we need it later
   int32 dotpos = file.rfind( "." );

   // Ok, if we're here we have to load the source.
   // should we force loading through Falcon Template Document parsing?
   bool bOldForceFtd = m_forceTemplate;
   if ( m_detectTemplate && ! m_forceTemplate )
   {
      if ( file.subString( dotpos ) == ".ftd" )
         m_forceTemplate = true;
   }

   // use the base load source routine
   // ask for detection, but default to falcon source
   // will throw on error
   std::auto_ptr<Stream> in( openResource( file, t_source ) );

   String modName;
   getModuleName( file, modName );

   Module *mod = 0;
   try {
      mod = loadSource( in.get(), file, modName );
      m_forceTemplate = bOldForceFtd;
      in->close();
   }
   catch (Error *)
   {
      // reset old forcing method
      m_forceTemplate = bOldForceFtd;
      throw;
   }

   return mod;
}


Module *ModuleLoader::loadSource( Stream *fin, const String &path, const String &name )
{
   Module *module;
   m_compileErrors = 0;

   // the temporary binary file for the pre-generated module
   Stream *temp_binary;

   module = new Module();
   module->name( name );
   module->path( path );

   m_compiler.reset();
   m_compiler.searchPath( getSearchPath() );

   if ( m_forceTemplate )
      m_compiler.parsingFtd( true );

   // the compiler can never throw
   if( ! m_compiler.compile( module, fin ) )
   {
      module->decref();
      throw m_compiler.detachErrors();
   }

   // we have compiled it. Now we need a file or a memory stream for
   // saving data.
   if ( m_compMemory )
   {
      temp_binary = new StringStream;
   }
   else {
      String tempFileName;
      Sys::_tempName( tempFileName );
      FileStream *tb= new FileStream;
      tb->create( tempFileName + "_1", Falcon::FileStream::e_aReadOnly | Falcon::FileStream::e_aUserWrite );
      if ( ! tb->good() )
      {
         int fserr = (int)tb->lastError();
         delete tb;
         module->decref();
         raiseError( e_file_output, tempFileName, fserr );
      }
      temp_binary = new StreamBuffer( tb );
   }

   GenCode codeOut( module );
   codeOut.generate( m_compiler.sourceTree() );

   // import the binary stream in the module;
   delete temp_binary;

   module->name( name );
   module->path( path );

   // if the base load source worked, save the result (if configured to do so).
   if ( m_saveModule )
   {
      URI tguri( path );
      fassert( tguri.isValid() );
      VFSProvider* vfs = Engine::getVFS( tguri.scheme() );
      fassert( vfs != 0 );

      // don't save on remote systems if saveRemote is false
      if ( vfs->protocol() == "file" || m_saveRemote )
      {
         tguri.pathElement().setExtension( "fam" );
         // Standard creations params are ok.
         Stream *temp_binary = vfs->create( tguri, VFSProvider::CParams() );
         if ( temp_binary == 0 || ! module->save( temp_binary ) )
         {
            if ( m_saveMandatory )
            {
               int fserr = (int) temp_binary->lastError();
               delete temp_binary;
               module->decref();
               raiseError( e_file_output, tguri.get(), fserr );
            }
         }

         delete temp_binary;
      }
   }

   return module;
}



}

/* end of flc_modloader.cpp */
