/*
   FALCON - The Falcon Programming Language.
   FILE: dir.cpp
   $Id: dir_ext.cpp,v 1.20 2007/08/11 13:08:29 jonnymind Exp $

   Directory management api.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: dom nov 7 2004
   Last modified because:

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
   In order to use this file in its compiled form, this source or
   part of it you have to read, understand and accept the conditions
   that are stated in the LICENSE file that comes boundled with this
   package.
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
#include <string.h>
#include "falcon_rtl_ext.h"

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

   // returns an array of 4 elemetns. ALWAYS.
   Item *name = vm->param(0);
   if ( name == 0 || ! name->isString() )
   {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).origin( e_orig_runtime ) ) );
      return;
   }

   CoreArray *parts = new CoreArray( vm, 4 );
   String *strName = name->asString();

   int32 pos = strName->length() -1;
   int32 last = pos;
   int state = 0; // scanning the extension.

   int32 ext_begin = -1, ext_end = -1;
   int32 fname_begin = -1, fname_end = -1;
   int32 fpath_begin = -1, fpath_end = -1;
   int32 unit_begin = -1, unit_end = -1;

   while ( pos >= 0 ) {
      char chr = strName->getCharAt(pos);
      switch ( state ) {
         case 0:  // extension?
            if ( chr == '.' && pos > 0 ) {
               ext_begin = pos + 1;
               ext_end = last;
               last = pos-1;
               state = 1;
            }
            else if ( chr == '/' || chr == ':' ) {
               fname_begin = pos + 1;
               fname_end = last;
               last = pos-1;
               state = chr == '/' ? 2 : 3;
            }
         break;

         case 1:
            if ( chr == '/' || chr == ':' ) {
               fname_begin = pos + 1;
               fname_end = last;
               // a "/.name" unix name?
               if ( fname_begin >= fname_end && ext_begin > 0 ) {
                  fname_begin = ext_begin-1;
                  fname_end = ext_end;
                  ext_begin = -1;
               }
               last = pos-1;
               state = chr == '/' ? 2 : 3;
            }
         break;

         case 2:
            if ( chr == ':' ) {
               fpath_begin = pos + 1;
               fpath_end = last;
               last = pos-1;
               state = 3;
            }
         break;

      }
      pos--;
   }
   pos = 0;
   switch( state ) {
      case 0: // never a single extension.
      case 1: fname_begin = pos; fname_end = last; break;
      case 2: fpath_begin = pos; fpath_end = last;
         if ( strName->getCharAt(fpath_begin) == '/' && fpath_end < fpath_begin )
            fpath_end = fpath_begin;
      break;
      case 3: unit_begin = pos; unit_end = last; break;
   }


   if ( ext_begin >= 0 ){
      if( ext_begin <= ext_end ) {
         String *ret = new GarbageString( vm, strName->subString( ext_begin, ext_end + 1 ) );
         parts->elements()[3] = ret;
      }
      else {
         parts->elements()[3] = new GarbageString( vm );
      }
   }
   else
      parts->elements()[3].setNil();

   if ( fname_begin >= 0 ){
      if( fname_begin <= fname_end ) {
         String *ret = new GarbageString( vm, strName->subString( fname_begin, fname_end + 1 ) );
         parts->elements()[2] = ret;
      }
      else {
         parts->elements()[2].setString( new GarbageString( vm ) );
      }
   }
   else
      parts->elements()[2].setNil();

   if ( fpath_begin >= 0 ){
      if( fpath_begin <= fpath_end ) {
         String *ret = new GarbageString( vm, strName->subString( fpath_begin, fpath_end + 1 ) );
         parts->elements()[1] = ret;
      }
      else {
         parts->elements()[1].setString( new GarbageString( vm ) );
      }
   }
   else
      parts->elements()[1].setNil();

   if ( unit_begin >= 0 ){
      if( unit_begin <= unit_end ) {
         String *ret = new GarbageString( vm, strName->subString( unit_begin, unit_end + 1 ) );
         parts->elements()[0] = ret;
      }
      else {
         parts->elements()[0].setString( new GarbageString( vm ) );
      }
   }
   else
      parts->elements()[0].setNil();

   parts->length( 4 );
   vm->retval( parts );
}

FALCON_FUNC  fileNameMerge ( ::Falcon::VMachine *vm )
{
   Item *unitspec = vm->param(0);
   Item *fname = 0;
   Item *fpath = 0;
   Item *fext = 0;

   if ( unitspec == 0 )
   {
      vm->retval( new GarbageString( vm ) );
      return;
   }

   if ( unitspec->isArray() )
   {
      CoreArray *array = unitspec->asArray();
      if( array->length() > 0 ) {
         unitspec = &(array->at( 0 ) );
      }
      if ( array->length() > 1 ) {
         fpath = &(array->at( 1 ) );
      }
      if ( array->length() > 2 ) {
         fname = &(array->at( 2 ) );
      }
      if ( array->length() > 3 ) {
         fext = &(array->at( 3 ) );
      }
   }
   else {
      fpath = vm->param(1);
      fname = vm->param(2);
      fext = vm->param(3);
   }

   GarbageString *buffer = new GarbageString( vm );

   if( unitspec != 0 && unitspec->isString() )
   {
      *buffer += *unitspec->asString() + ":";
   }

   if( fpath != 0 && fpath->isString() )
   {
      *buffer += *fpath->asString();
      if ( buffer->getCharAt( buffer->length()-1 ) != '/' )
         *buffer += '/';
   }

   if( fname != 0 && fname->isString() )
   {
      *buffer += *fname->asString();
   }

   if( fext != 0 && fext->isString() )
   {
      *buffer += "." + *fext->asString();
   }

   vm->retval( buffer );
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
   String *strName = name->asString();

   int32 fsError;
   if( ! Sys::fal_mkdir( *strName, fsError ) ) {
      vm->raiseModError( new IoError( ErrorParam( 1011, __LINE__ ).
         origin( e_orig_runtime ).desc( "Cannot create directory" ).extra( *strName ).
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
         origin( e_orig_runtime ).desc( "Cannot move target file" ).extra( *strName + "->" + *strDest ).
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

}
}


/* end of dir.cpp */
