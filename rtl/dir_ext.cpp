/*
   FALCON - The Falcon Programming Language.
   FILE: dir.cpp

   Directory management api.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: dom nov 7 2004

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

/** \file
   Directory management api
*/

#include <falcon/setup.h>
#include <falcon/module.h>
#include <falcon/sys.h>
#include <falcon/item.h>
#include <falcon/vm.h>
#include <falcon/string.h>
#include <falcon/filestat.h>
#include <falcon/dir_sys.h>
#include <falcon/memory.h>
#include <falcon/fassert.h>
#include <falcon/uri.h>

#include "falcon_rtl_ext.h"
#include "rtl_messages.h"

#include <string.h>


namespace Falcon {

namespace Ext {

static void Stats_to_object( VMachine *vm, const FileStat &fstats, CoreObject *self )
{
   self->setProperty( "size", fstats.m_size );
   self->setProperty( "type", (int64) fstats.m_type );
   self->setProperty( "owner", (int64) fstats.m_owner );
   self->setProperty( "group", (int64) fstats.m_group );
   self->setProperty( "attribs", (int64) fstats.m_attribs );
   self->setProperty( "access", (int64) fstats.m_access );

   // create the timestamps
   Item *ts_class = vm->findWKI( "TimeStamp" );
   //if we wrote the std module, can't be zero.
   fassert( ts_class != 0 );
   CoreObject *timestamp= ts_class->asClass()->createInstance();

   fassert( timestamp != 0 ); // as it should be declared around here...
   timestamp->setUserData( new TimeStamp );
   static_cast<TimeStamp *>(timestamp->getUserData())->copy( *fstats.m_mtime );
   self->setProperty( "mtime", timestamp );

   timestamp= ts_class->asClass()->createInstance();
   timestamp->setUserData( new TimeStamp );
   static_cast<TimeStamp *>(timestamp->getUserData())->copy( *fstats.m_ctime );
   self->setProperty( "ctime", timestamp );

   timestamp= ts_class->asClass()->createInstance();
   timestamp->setUserData( new TimeStamp );
   static_cast<TimeStamp *>(timestamp->getUserData())->copy( *fstats.m_atime );
   self->setProperty( "atime", timestamp );
}

FALCON_FUNC FileReadStats( ::Falcon::VMachine *vm )
{
   Item *name = vm->param(0);

   // if the name is not given, we expect a readStats to be called later on
   if ( name == 0 )
      return;

   if ( ! name->isString() )
   {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).origin( e_orig_runtime ) ) );
      return;
   }

   // create the timestamps
   Item *fs_class = vm->findWKI( "FileStat" );
   //if we wrote the std module, can't be zero.
   fassert( fs_class != 0 );
   FileStat fstats;
   CoreObject *self = fs_class->asClass()->createInstance();
   if ( ! Sys::fal_stats( *name->asString(), fstats ) )
   {
      String fname =  *name->asString();
      vm->raiseModError( new IoError( ErrorParam( 1001, __LINE__ ).
         origin( e_orig_runtime ).desc( "Cannot read stats for file" ).extra( fname ).
         sysError( (uint32) Sys::_lastError() ) ) );
      return;
   }

   Stats_to_object( vm, fstats, self );
   vm->retval( self );
}

FALCON_FUNC FileStat_readStats ( ::Falcon::VMachine *vm )
{
   Item *name = vm->param(0);

   if ( name == 0 || ! name->isString() )
   {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).origin( e_orig_runtime ) ) );
      return;
   }

   FileStat fstats;
   CoreObject *self = vm->self().asObject();
   if ( ! Sys::fal_stats( *name->asString(), fstats ) )
   {
      vm->retval( 0 );
   }
   else
   {
      Stats_to_object( vm, fstats, self );
      vm->retval( 1 );
   }
}

FALCON_FUNC  fileType( ::Falcon::VMachine *vm )
{
   Item *name = vm->param(0);

   if ( name == 0 || ! name->isString() )
   {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).origin( e_orig_runtime ) ) );
      return;
   }

   FileStat::e_fileType type;

   Sys::fal_fileType( *name->asString(), type );
   // will already be -1 if not found
   vm->retval( type );
}


FALCON_FUNC  dirReadLink( ::Falcon::VMachine *vm )
{
   Item *name = vm->param(0);

   if ( name == 0 || ! name->isString() )
   {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).origin( e_orig_runtime ) ) );
      return;
   }

   String temp;
   if ( ! Sys::fal_readlink( *name->asString(), temp ) ) {
      vm->retnil();
   }
   else {
      String *ret = new GarbageString( vm );
      ret->bufferize( temp );
      vm->retval( ret );
   }
}

FALCON_FUNC  dirMakeLink( ::Falcon::VMachine *vm )
{
   Item *name = vm->param(0);
   Item *dest = vm->param(1);

   if ( name == 0 || ! name->isString() || dest == 0 || ! dest->isString() )
   {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).origin( e_orig_runtime ) ) );
      return;
   }

   if ( ! Sys::fal_readlink( *name->asString(), *dest->asString() ) ) {
      vm->retval( 0 );
   }
   else {
      vm->retval( 1 );
   }
}

FALCON_FUNC  fileNameSplit ( ::Falcon::VMachine *vm )
{
   // a filename is always in this format:
   // [DISK_SPEC or SERVER:][/path/to/file/]file.part[.ext]

   // returns an array of 4 elements. ALWAYS.
   Item *name = vm->param(0);
   if ( name == 0 || ! name->isString() )
   {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).origin( e_orig_runtime ) ) );
      return;
   }

   Path path( *name->asString() );
   if ( ! path.isValid() )
   {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).
         origin( e_orig_runtime ).
         extra( vm->moduleString( msg::rtl_invalid_path ) ) ) );
      return;
   }

   CoreArray *parts = new CoreArray( vm, 4 );
   parts->length( 4 );
   String *res = new GarbageString( vm );
   String *loc = new GarbageString( vm );
   String *file = new GarbageString( vm );
   String *ext = new GarbageString( vm );

   path.split( *res, *loc, *file, *ext );
   parts->at(0) = res;
   parts->at(1) = loc;
   parts->at(2) = file;
   parts->at(3) = ext;

   vm->retval( parts );
}

FALCON_FUNC  fileNameMerge ( ::Falcon::VMachine *vm )
{
   const String *unitspec = 0;
   const String *fname = 0;
   const String *fpath = 0;
   const String *fext = 0;

   String sDummy;
   Item *p0 = vm->param(0);

   if ( p0 != 0 && p0->isArray() )
   {
      const CoreArray &array = *p0->asArray();
      if( array.length() >= 0 && array[0].isString() ) {
         unitspec = array[0].asString();
      }
      else
         unitspec = &sDummy;

      if( array.length() >= 1 && array[1].isString() ) {
         fpath = array[1].asString();
      }
      else
         fpath = &sDummy;

      if( array.length() >= 2 && array[2].isString() ) {
         fname = array[2].asString();
      }
      else
         fname = &sDummy;

      if( array.length() >= 3 && array[3].isString() ) {
         fext = array[3].asString();
      }
      else
         fext = &sDummy;

   }
   else {
      Item *p1 = vm->param(1);
      Item *p2 = vm->param(2);
      Item *p3 = vm->param(3);

      unitspec = p0 != 0 && p0->isString() ? p0->asString() : &sDummy;
      fpath = p1 != 0 && p1->isString() ? p1->asString() : &sDummy;
      fname = p2 != 0 && p2->isString() ? p2->asString() : &sDummy;
      fext = p3 != 0 && p3->isString() ? p3->asString() : &sDummy;
   }

   Path p;
   p.join( *unitspec, *fpath, *fname, *fext );
   if ( ! p.isValid() )
   {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).
         origin( e_orig_runtime ).
         extra( vm->moduleString( msg::rtl_invalid_path ) ) ) );
      return;
   }

   vm->retval( new GarbageString( vm, p.get() ) );
}


FALCON_FUNC  fileName ( ::Falcon::VMachine *vm )
{
   Item *filename = vm->param(0);
   if ( filename == 0 || ! filename->isString() )
   {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).origin( e_orig_runtime ) ) );
      return;
   }

   String *name = filename->asString();
   int32 len = name->length()-1;
   int32 pos = len;
   while( pos >= 0 )
   {
      if ( name->getCharAt( pos ) == '/' )
      {
         vm->retval( new GarbageString( vm, *name, pos + 1 ) );
         return;
      }
      pos--;
   }

   // shallow copy
   vm->retval( *filename );
}

FALCON_FUNC  filePath ( ::Falcon::VMachine *vm )
{
   Item *filename = vm->param(0);
   if ( filename == 0 || ! filename->isString() )
   {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).origin( e_orig_runtime ) ) );
      return;
   }


   String *name = filename->asString();
   int32 len = name->length();
   int32 pos = len-1;

   while( pos > 0 ) {
      if ( name->getCharAt( pos ) == '/' ) {
         vm->retval( new GarbageString( vm, *name, 0, pos ) );
         return;
      }
      pos--;
   }

   if ( name->getCharAt( pos ) == '/' )
      vm->retval( new GarbageString( vm, "/" ) );
   else
      vm->retval( new GarbageString( vm ) );
}

FALCON_FUNC  DirectoryOpen ( ::Falcon::VMachine *vm )
{
   Item *name = vm->param(0);
   if ( name == 0 || ! name->isString() )
   {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).origin( e_orig_runtime ) ) );
      return;
   }

   int32 fsError;
   DirEntry *dir = Sys::fal_openDir( *name->asString(), fsError );

   if( dir != 0 ) {
      Item *dir_class = vm->findWKI( "Directory" );
      //if we wrote the std module, can't be zero.
      fassert( dir_class != 0 );
      CoreObject *self = dir_class->asClass()->createInstance();
      self->setUserData( dir );
      vm->retval( self );
   }
   else {
      vm->raiseModError( new IoError( ErrorParam( 1010, __LINE__ ).
         origin( e_orig_runtime ).desc( "Can't open directory" ).extra( *name->asString() ).
         sysError( (uint32) Sys::_lastError() ) ) );
   }
}

FALCON_FUNC  Directory_read ( ::Falcon::VMachine *vm )
{
   DirEntry *dir = static_cast<DirEntry *>(vm->self().asObject()->getUserData());

   String reply;
   if ( dir->read( reply ) ) {
      String *ret = new GarbageString( vm );
      ret->bufferize( reply );
      vm->retval( ret );
   }
   else {
      if ( dir->lastError() != 0 ) {
         vm->raiseModError( new IoError( ErrorParam( 1010, __LINE__ ).
            origin( e_orig_runtime ).desc( "Can't read directory" ).
            sysError( (uint32) Sys::_lastError() ) ) );
      }
      vm->retnil();
   }
}

FALCON_FUNC  Directory_close ( ::Falcon::VMachine *vm )
{
   DirEntry *dir = static_cast<DirEntry *>(vm->self().asObject()->getUserData());
   dir->close();
   if ( dir->lastError() != 0 ) {
      vm->raiseModError( new IoError( ErrorParam( 1010, __LINE__ ).
         origin( e_orig_runtime ).desc( "Can't close directory" ).
         sysError( (uint32) Sys::_lastError() ) ) );
   }
}

FALCON_FUNC  Directory_error( ::Falcon::VMachine *vm )
{
   DirEntry *dir = static_cast<DirEntry *>(vm->self().asObject()->getUserData());
   vm->retval( (int)dir->lastError() );
}


FALCON_FUNC  dirMake ( ::Falcon::VMachine *vm )
{
   Item *name = vm->param(0);
   if ( name == 0 || ! name->isString() )
   {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).origin( e_orig_runtime ) ) );
      return;
   }
   const String &strName = *name->asString();
   bool descend = vm->param(1) == 0 ? false : vm->param(1)->isTrue();

   int32 fsError = 0;
   if ( descend )
   {
      // find /.. sequences
      uint32 pos = strName.find( "/" );
      while( true )
      {
         String strPath( strName, 0, pos );

         // stat the file
         FileStat fstats;
         // if the file exists...
         if ( (! Sys::fal_stats( strPath, fstats )) ||
              fstats.m_type != FileStat::t_dir )
         {
            // if it's not a directory, try to create the directory.
            if ( ! Sys::fal_mkdir( strPath, fsError ) )
               break;
         }

         // last loop?
         if ( pos == String::npos )
            break;

         pos = strName.find( "/", pos + 1 );
       }
   }
   else
   {
      // Just one try; succeed or fail
      Sys::fal_mkdir( strName, fsError );
   }

   if ( fsError != 0 )
   {
      vm->raiseModError( new IoError( ErrorParam( 1011, __LINE__ ).
         origin( e_orig_runtime ).desc( "Cannot create directory" ).extra( strName ).
         sysError( (uint32) Sys::_lastError() ) ) );
   }
}


FALCON_FUNC  dirRemove ( ::Falcon::VMachine *vm )
{
   Item *name = vm->param(0);
   if ( name == 0 || ! name->isString() )
   {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).origin( e_orig_runtime ) ) );
      return;
   }
   String *strName = name->asString();

   int32 fsError;
   if( ! Sys::fal_rmdir( *strName, fsError ) ) {
      vm->raiseModError( new IoError( ErrorParam( 1012, __LINE__ ).
         origin( e_orig_runtime ).desc( "Cannot remove directory" ).extra( *strName ).
         sysError( (uint32) Sys::_lastError() ) ) );
   }
}


FALCON_FUNC  dirChange ( ::Falcon::VMachine *vm )
{
   Item *name = vm->param(0);
   if ( name == 0 || ! name->isString() )
   {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).origin( e_orig_runtime ) ) );
      return;
   }
   String *strName = name->asString();

   int32 fsError;
   if( ! Sys::fal_chdir( *strName, fsError ) ) {
      vm->raiseModError( new IoError( ErrorParam( 1013, __LINE__ ).
         origin( e_orig_runtime ).desc( "Cannot change working directory" ).extra( *strName ).
         sysError( (uint32) Sys::_lastError() ) ) );
   }
   else
      vm->retnil();
}

FALCON_FUNC  dirCurrent ( ::Falcon::VMachine *vm )
{
   int32 fsError;
   String *ret = new GarbageString( vm );
   if( ! Sys::fal_getcwd( *ret, fsError ) ) {
      delete ret;
      vm->raiseModError( new IoError( ErrorParam( 1014, __LINE__ ).
         origin( e_orig_runtime ).desc( "Cannot read current working directory" ).
         sysError( (uint32) Sys::_lastError() ) ) );
   }
   else
      vm->retval( ret );
}



FALCON_FUNC  fileRemove ( ::Falcon::VMachine *vm )
{
   Item *name = vm->param(0);
   if ( name == 0 || ! name->isString() )
   {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).
         origin( e_orig_runtime ) ) );
      return;
   }
   String *strName = name->asString();

   int32 fsError;
   if( ! Sys::fal_unlink( *strName, fsError ) ) {
      vm->raiseModError( new IoError( ErrorParam( 1015, __LINE__ ).
         origin( e_orig_runtime ).desc( "Cannot remove target file" ).extra( *strName ).
         sysError( (uint32) Sys::_lastError() ) ) );
   }
}


FALCON_FUNC  fileMove ( ::Falcon::VMachine *vm )
{
   Item *name = vm->param(0);
   Item *dest = vm->param(1);

   if ( name == 0 || ! name->isString() || dest == 0 || ! dest->isString() )
   {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).
         origin( e_orig_runtime ) ) );
      return;
   }

   String *strName = name->asString();
   String *strDest = dest->asString();

   int32 fsError;
   if( ! Sys::fal_move( *strName, *strDest, fsError ) ) {
      vm->raiseModError( new IoError( ErrorParam( 1016, __LINE__ ).
         origin( e_orig_runtime ).desc( "Cannot move target file" ).extra( *strName + " -> " + *strDest ).
         sysError( (uint32) Sys::_lastError() ) ) );
   }
}

FALCON_FUNC  fileChmod ( ::Falcon::VMachine *vm )
{
   Item *name = vm->param(0);
   Item *mode = vm->param(1);

   if ( name == 0 || ! name->isString() || mode == 0 || ! mode->isOrdinal() )
   {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).
      origin( e_orig_runtime ) ) );
      return;
   }

   if( ! Sys::fal_chmod( *name->asString(), (uint32) mode->forceInteger() ) )
   {
      vm->raiseModError( new IoError( ErrorParam( 1016, __LINE__ ).
         origin( e_orig_runtime ).desc( "Cannot change target file mode" ).extra( *name->asString() ).
         sysError( (uint32) Sys::_lastError() ) ) );
   }
}

FALCON_FUNC  fileChown ( ::Falcon::VMachine *vm )
{
   Item *name = vm->param(0);
   Item *mode = vm->param(1);

   if ( name == 0 || ! name->isString() || mode == 0 || ! mode->isOrdinal() )
   {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).
         origin( e_orig_runtime ) ) );
      return;
   }

   if( Sys::fal_chown( *name->asString(), (int32) mode->forceInteger() ) )
   {
      vm->raiseModError( new IoError( ErrorParam( 1017, __LINE__ ).
         origin( e_orig_runtime ).desc( "Cannot change target file owner" ).extra( *name->asString() ).
         sysError( (uint32) Sys::_lastError() ) ) );
   }
}

FALCON_FUNC  fileChgroup ( ::Falcon::VMachine *vm )
{
   Item *name = vm->param(0);
   Item *mode = vm->param(1);

   if ( name == 0 || ! name->isString() || mode == 0 || ! mode->isOrdinal() )
   {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).
         origin( e_orig_runtime ) ) );
      return;
   }

   if( Sys::fal_chgrp( *name->asString(), (int32) mode->forceInteger() ) )
   {
      vm->raiseModError( new IoError( ErrorParam( 1018, __LINE__ ).
         origin( e_orig_runtime ).desc( "Cannot change target file owner" ).extra( *name->asString() ).
         sysError( (uint32) Sys::_lastError() ) ) );
   }
}

//====================================================
// (reflexive) Class path
//

/*# @class Path
   @brief Interface to local filesystem path definition.
   @optparam path The path that will be used as initial path.
   @raise ParamError in case the inital path is malformed.

   This class offers an object oriented interface to access
   path elements given a complete path, or to build a path from its elements.
*/

/*# @property unit Path
   @brief Unit specificator.
   @raise ParamError if assigned to a value that makes the path invalid.

   This is the unit specificator (disk name) used in some filesystems.
   It is separated by the rest of the path via a ":". According to
   RFC 3986 it always starts with a "/", which is automatically added
   if absent.
*/

/*# @property location Path
   @brief Location specificator.
   @raise ParamError if assigned to a value that makes the path invalid.

   This is the "path to file". It can start with a "/" or not; if
   it starts with a "/" it is considered absolute.
*/

/*# @property file Path
   @brief File part.
   @raise ParamError if assigned to a value that makes the path invalid.

   This is the part of the path that identifies an element in a directory.
   It includes everything after the last "/" path separator.
*/

/*# @property filename Path
   @brief File name part.
   @raise ParamError if assigned to a value that makes the path invalid.

   This element coresponds to the first part of the file element, if it is
   divided into a filename and an extension by a "." dot.
*/

/*# @property extension Path
   @brief File extension part.
   @raise ParamError if assigned to a value that makes the path invalid.

   This element coresponds to the first last of the file element, if it is
   divided into a filename and an extension by a "." dot.
*/

/*# @property path Path
   @brief Complete path.
   @raise ParamError if assigned to a value that makes the path invalid.

   This is the complete path referred by this object.
*/

class PathCarrier: public UserData
{
public:
   Path m_path;
   String m_unit;
   String m_location;
   String m_file;
   String m_ext;
   String m_filename;

   PathCarrier() {}

   PathCarrier( const PathCarrier &other ):
      m_path( other.m_path )
      {}

   virtual void getProperty( VMachine *vm, const String &propName, Item &prop );
   virtual void setProperty( VMachine *vm, const String &propName, Item &prop );
   virtual bool isReflective() const;
   virtual UserData *clone() const;
};

bool PathCarrier::isReflective() const
{
   return true;
}

UserData *PathCarrier::clone() const
{
   return new PathCarrier( *this );
}

void PathCarrier::setProperty( VMachine *vm, const String &propName, Item &prop )
{
   if( ! prop.isString() )
   {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params ).
         origin( e_orig_runtime ).extra( "S" ) ) );
      return;
   }

   if ( propName == "unit" )
   {
      m_path.setResource( *prop.asString() );
   }
   else if ( propName == "location" )
   {
      m_path.setLocation( *prop.asString() );
   }
   else if ( propName == "file" )
   {
      m_path.setFile( *prop.asString() );
   }
   else if ( propName == "filename" )
   {
      m_path.setFilename( *prop.asString() );
   }
   else if ( propName == "extension" )
   {
      m_path.setExtension( *prop.asString() );
   }
   else if ( propName == "path" )
   {
      m_path.set( *prop.asString() );
   }

   if ( ! m_path.isValid() )
   {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params ).
         origin( e_orig_runtime ).
         extra( vm->moduleString( msg::rtl_invalid_path ) ) ) );
   }
}


void PathCarrier::getProperty( VMachine *vm, const String &propName, Item &prop )
{
   if ( ! m_path.isValid() )
   {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params ).
         origin( e_orig_runtime ).
         extra( vm->moduleString( msg::rtl_invalid_path ) ) ) );
   }

   String *item;
   if ( propName == "unit" )
   {
      m_path.getResource( m_unit );
      item = &m_unit;
   }
   else if ( propName == "location" )
   {
      m_path.getLocation( m_location );
      item = &m_location;
   }
   else if ( propName == "file" )
   {
      m_path.getFile( m_file );
      item = &m_file;
   }
   else if ( propName == "filename" )
   {
      m_path.getFilename( m_filename );
      item = &m_filename;
   }
   else if ( propName == "extension" )
   {
      m_path.getExtension( m_ext );
      item = &m_ext;
   }
   else if ( propName == "path" )
   {
      item = const_cast<String *>( &m_path.get() );
   }

   prop = item;
}


/*# @init Path
   @brief Constructor for the Path class.
   @raise ParamError in case the inital path is malformed.

   Builds the path object, optionally using the given parameter
   as a complete path constructor.

   If the parameter is an array, it must have at least four
   string elements, and it will be used to build the path from
   its constituents. In example:

   @code
      unit = "C"
      location = "/a/path/to"
      file = "somefile"
      ext = "anext"
      p = Path( [ unit, location, file, ext ] )
   @endocde

   @b nil can be passed if some part of the specification is not used.

   @note Use the fileNameMerge() function to simply merge elements of a path
   specification into a string.
   @see fileNameMerge
*/

FALCON_FUNC  Path_init ( ::Falcon::VMachine *vm )
{
   Item *p0 = vm->param(0);

   if ( p0 == 0 || ( ! p0->isString() && ! p0->isArray() ) )
   {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).
         origin( e_orig_runtime ).extra( "S|A" ) ) );
      return;
   }

   // we need anyhow a carrier.
   PathCarrier *carrier = new PathCarrier;
   CoreObject *self = vm->self().asObject();
   self->setUserData( carrier );

   if ( p0->isString() )
   {
      carrier->m_path.set( *p0->asString() );
   }
   else {
      const String *unitspec = 0;
      const String *fname = 0;
      const String *fpath = 0;
      const String *fext = 0;

      String sDummy;

      const CoreArray &array = *p0->asArray();
      if( array.length() >= 0 && array[0].isString() )
         unitspec = array[0].asString();
      else
         unitspec = &sDummy;

      if( array.length() >= 1 && array[1].isString() )
         fpath = array[1].asString();
      else
         fpath = &sDummy;

      if( array.length() >= 2 && array[2].isString() )
         fname = array[2].asString();
      else
         fname = &sDummy;

      if( array.length() >= 3 && array[3].isString() )
         fext = array[3].asString();
      else
         fext = &sDummy;

      carrier->m_path.join( *unitspec, *fpath, *fname, *fext );
   }

   if ( ! carrier->m_path.isValid() )
   {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).
         origin( e_orig_runtime ).
         extra( vm->moduleString( msg::rtl_invalid_path ) ) ) );
   }

}

}
}

/* end of dir.cpp */
