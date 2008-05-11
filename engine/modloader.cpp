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
#include <falcon/fstream.h>
#include <falcon/module.h>
#include <falcon/dll.h>
#include <falcon/fstream.h>
#include <falcon/sys.h>
#include <falcon/pcodes.h>

namespace Falcon {



ModuleLoader::ModuleLoader( const String &path ):
   m_errhand(0),
   m_acceptSources( false )
{
   m_path.deletor( string_deletor );
   setSearchPath( path );
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

ModuleLoader::~ModuleLoader()
{
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

Stream *ModuleLoader::openResource( const String &path, t_filetype )
{
   FileStream *in = new FileStream;
   if ( ! in->open( path, FileStream::e_omReadOnly ) )
   {
      raiseError( e_open_file, path );
      delete in;
      return 0;
   }

   return in;
}

ModuleLoader::t_filetype ModuleLoader::fileType( const String &path )
{
   String ext = path.subString( path.rfind( "." ) );

   FileStream in;

   if ( ext == ".fal" || ext == ".ftd" ) {
      in.open( path, FileStream::e_omReadOnly );
      if ( ! in.good() )
         return t_none;
      return t_source;
   }
   else if ( ext == DllLoader::dllExt() || ext == ".fam" )
   {
      in.open( path, FileStream::e_omReadOnly );
      char ch1, ch2, ch3, ch4;
      if ( ! in.good() )
         return t_none;
      in.read( &ch1, 1 );
      in.read( &ch2, 1 );
      in.read( &ch3, 1 );
      in.read( &ch4, 1 );
      in.close();
      if ( ext == DllLoader::dllExt() ) {
         return DllLoader::isDllMark( ch1, ch2 ) ? t_binmod : t_none;
      }
      else {
         if(ch1 =='F' && ch2 =='M') {
            // verify if version/subversion is accepted.
            if( ch3 == FALCON_PCODE_VERSION && ch4 == FALCON_PCODE_MINOR )
               return t_vmmod;
         }
         return t_none;
      }
   }

   return t_none;
}



ModuleLoader::t_filetype
      ModuleLoader::scanForFile( const String &name, bool isPath,
      ModuleLoader::t_filetype scanForType, String &found, bool accSrc )
{
   t_filetype tf = t_none;
   String path_name;
   String expName = name;

   // expand "." names into "/"
   uint32 pos = expName.find( "." );
   while( pos != String::npos )
   {
      expName.setCharAt( pos, '/' );
      pos = expName.find( ".", pos + 1 );
   }

   ListElement *path_elem = m_path.begin();
   while ( tf == t_none && path_elem != 0 )
   {
      String *pathp = (String *) path_elem->data();

      // scanning this path:
      if ( pathp->getCharAt( pathp->length() ) != '/' )
         path_name = *pathp + "/";
      else
         path_name = *pathp;


      // if it's a direct path, we must not add an extension.
      if ( isPath )
      {
         found = path_name + expName;
         tf = fileType( found );
      }
      else {
         // first try to search for the DLL
         found = path_name + expName + DllLoader::dllExt();
         tf = fileType( found );

         // then try for the source, if allowed
         if ( tf == t_none && accSrc )
         {
            found = path_name + expName + ".fal";
            tf = fileType( found );
         }

         // then try for the source, if allowed
         if ( tf == t_none && accSrc )
         {
            found = path_name + expName + ".ftd";
            tf = fileType( found );
         }

         // and then for the fam
         if ( tf == t_none )
         {
            found = path_name + expName + ".fam";
            tf = fileType( found );
         }
      }

      // if the path is not the one we want, reject it.
      if( scanForType != t_none && tf != scanForType )
         tf = t_none;

      // try again.
      path_elem = path_elem->next();
   }

   // return what we've found
   return tf;
}

Module *ModuleLoader::loadName( const String &module_name, const String &parent_name )
{
   String file_path;
   String nmodName;

   // prevent doing a crashy thing.
   if ( module_name.length() == 0 )
      return 0;

   if ( module_name.getCharAt(0) == '.' )
   {
      // notation .name
      if ( parent_name.size() == 0 )
         nmodName = module_name.subString( 1 );
      else {
         // remove last part of parent name
         uint32 posDot = parent_name.rfind( "." );
         // are there no dot? -- we're at root elements
         if ( posDot == String::npos )
            nmodName = module_name.subString( 1 );
         else
            nmodName = parent_name.subString( 0, posDot ) + module_name; // "." is included.
      }
   }
   else if ( module_name.find( "self." ) == 0 )
   {
      if ( parent_name.size() == 0 )
         nmodName = module_name.subString( 5 );
      else
         nmodName = parent_name + "." + module_name.subString( 5 );
   }
   else
      nmodName = module_name;

   t_filetype type = scanForFile( nmodName, false, t_none, file_path, m_acceptSources );

   Module *mod;
   switch( type )
   {
   case t_source: mod = loadSource( file_path ); break;
   case t_vmmod: mod = loadModule( file_path ); break;
   case t_binmod: mod = loadBinaryModule( file_path ); break;

   default:
      // we have not been able to find it.
      raiseError( e_nofile, nmodName );
      return 0;
   }

   if ( mod != 0 )
	{
      mod->name( nmodName );
      mod->path( file_path );
      mod->addMain();

      // should we set a language table?
      if ( m_language != "" && mod->language() != m_language )
      {
         loadLanguageTable( mod, m_language );
      }
	}

	// if the mod is 0, the load function has already raised the right error.
   return mod;
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
   // try to open the required file table.
   FileStream fsin;
   if ( ! fsin.open( file_path, FileStream::e_omReadOnly, FileStream::e_smShareRead ) )
      return false;

   // check if this is a regular tab file.
   char buf[16];
   buf[6] = 0;
   if ( fsin.read( buf, 5 ) != 5 || String( "TLTAB" ) != buf )
   {
      return false;
   }

   uint16 endianity;
   if( fsin.read( &endianity, 2 ) != 2 )
      return false;

   bool sameEndianity = endianity == 0xFBFC;

   // read the language table index.
   int32 sizeField;
   if( fsin.read( &sizeField, 4 ) != 4 )
      return false;

   int32 tableSize = xendianity( sameEndianity, sizeField );
   int32 tablePos = -1;
   for( int32 i = 0; i < tableSize; i++ )
   {
      // read language code and position in file
      if( fsin.read( buf, 5 ) != 5 ||
          fsin.read( &sizeField, 4 ) != 4 )
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

   fsin.seekBegin( tableSize * 9 + 5 + 2 + 4  + tablePos );

   // read the number of strings to be decoded.
   if( fsin.read( &sizeField, 4 ) != 4 )
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
      if( fsin.read( &sizeField, 4 ) != 4 )
         break;
      int32 stringID = xendianity( sameEndianity, sizeField );
      if ( stringID < 0 || stringID >= mod->stringTable().size() )
         break;

      // read the string size
      if( fsin.read( &sizeField, 4 ) != 4 )
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
      if( fsin.read( memBuf, stringSize ) != stringSize )
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

Module *ModuleLoader::loadFile( const String &module_path, t_filetype type, bool scan )
{
   String file_path;

   if ( type == t_none || type == t_defaultSource )
   {
      t_filetype t_orig = type;

      if ( scan )
         type = scanForFile( module_path, true, t_none, file_path, m_acceptSources );
      else {
         type = fileType( module_path );
         file_path = module_path;
      }

      // if the type is unknown, should we default so source?
      if ( type == t_none && t_orig == t_defaultSource )
         type = t_source;
   }

   Module *mod;
   switch( type )
   {
   case t_source: mod = loadSource( file_path ); break;
   case t_vmmod: mod = loadModule( file_path ); break;
   case t_binmod: mod = loadBinaryModule( file_path ); break;

   default:
      // we have not been able to find it.
      raiseError( e_nofile, module_path );
      return 0;
   }

   if ( mod != 0 )
	{
      String modName;
      getModuleName( file_path, modName );
      mod->name( modName );
      mod->path( file_path );
      mod->addMain();
      if ( m_language != "" && mod->language() != m_language )
      {
         loadLanguageTable( mod, m_language );
      }
	}

	// if the mod is 0, the load function has already raised the right error.

   return mod;
}


Module *ModuleLoader::loadModule( const String &path )
{
   Stream *in = openResource( path, t_vmmod );

   if ( in == 0 )
   {
      return 0;
   }

   Module *mod = loadModule( in );
   in->close();
   delete in;

   if ( mod == 0 ) {
      raiseError( e_invformat, path );
   }
   else {
      String modName;
      getModuleName( path, modName );
      mod->name( modName );
      mod->path( path );
      mod->addMain();
      if ( m_language != "" && mod->language() != m_language )
      {
         loadLanguageTable( mod, m_language );
      }
   }

   return mod;
}



Module *ModuleLoader::loadSource( const String &path )
{
   Stream *in = openResource( path, t_source );

   if ( in == 0 )
   {
      return 0;
   }

   Module *mod = loadSource( in, path );
   in->close();
   delete in;

   // ! don't raise an error if mod == 0; someone has already done it.
   if ( mod != 0 )
   {
      String modName;
      getModuleName( path, modName );
      mod->name( modName );
      mod->path( path );
      mod->addMain();
      if ( m_language != "" && mod->language() != m_language )
      {
         loadLanguageTable( mod, m_language );
      }
   }

   return mod;
}

Module *ModuleLoader::loadSource( Stream *in, const String &path )
{
   raiseError( e_loader_unsupported, "loadSource" );
   return 0;
}



Module *ModuleLoader::loadBinaryModule( const String &path )
{
   DllLoader dll;

   if ( ! dll.open( path ) )
   {
      String error;
      dll.getErrorDescription( error );
      raiseError( e_binload, path + ":" + error );
      return 0;
   }

   DllFunc dlfunc = dll.getSymbol( "falcon_module_init" );
   ext_mod_init func = (ext_mod_init) dlfunc.data();

   if ( func == 0 )
   {
      raiseError( e_binstartup, path );
      return 0;
   }

   // creating an instance here means we're getting static data
   EngineData data;
   Module *mod = func( data );

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

   if(c1 =='F' && c2 =='M') {
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

   raiseError( e_modver );
   return 0;
}

Module *ModuleLoader::loadModule_ver_1_0( Stream *in )
{
   Module *mod = new Module();
   if ( ! mod->load( in, true ) )
   {
      raiseError( e_modformat );
   }
   return mod;
}


void  ModuleLoader::raiseError( int code, const String &expl )
{
   if ( m_errhand != 0 )
   {
      Error *error = new IoError( ErrorParam( code ).extra( expl ).origin( e_orig_loader ).
         module( "core" ).
         module( "(Module loader)" )
       );
      m_errhand->handleError( error );
      error->decref();
   }
}

}

/* end of flc_modloader.cpp */
