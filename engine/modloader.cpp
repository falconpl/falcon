/*
   FALCON - The Falcon Programming Language.
   FILE: flc_modloader.cpp

   Module loader
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: 2004-08-20
   Last modified because:

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
   In order to use this file in its compiled form, this source or
   part of it you have to read, understand and accept the conditions
   that are stated in the LICENSE file that comes boundled with this
   package.
*/

#include <falcon/setup.h>
#include <falcon/modloader.h>
#include <falcon/fstream.h>
#include <falcon/module.h>
#include <falcon/dll.h>
#include <falcon/fstream.h>
#include <falcon/sys.h>

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
         m_path.pushBack( tmp );
         break;
      }

      if ( pos1 == -1 ) {
         tmp = new String( path, pos );
         m_path.pushBack( tmp );
         break;
      }

      if ( pos1 > pos ) {
         tmp = new String( path, pos, pos1 );
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
      char ch1, ch2;
      if ( ! in.good() )
         return t_none;
      in.read( &ch1, 1 );
      in.read( &ch2, 1 );
      in.close();
      if ( ext == DllLoader::dllExt() ) {
         return DllLoader::isDllMark( ch1, ch2 ) ? t_binmod : t_none;
      }
      else {
         if(ch1 =='F' && ch2 =='M') {
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

   if ( module_name.find( "self." ) == 0 )
   {
      if ( parent_name.size() == 0 )
         nmodName = module_name;
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
	}

	// if the mod is 0, the load function has already raised the right error.

   return mod;
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
   // for now we can load only format 1.0

   if( c1 == 1 && c2 == 0 ) {
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
