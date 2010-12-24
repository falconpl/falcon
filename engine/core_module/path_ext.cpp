/*
   FALCON - The Falcon Programming Language.
   FILE: dir.cpp

   Directory management api.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Wed, 28 Jan 2009 07:53:27 -0800

   -------------------------------------------------------------------
   (C) Copyright 2009: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

/*#
   @beginmodule core
*/

/** \file
   Path - script interface API
*/

#include <falcon/setup.h>
#include <falcon/fassert.h>
#include <falcon/path.h>
#include <falcon/eng_messages.h>
#include <falcon/string.h>
#include <falcon/crobject.h>
#include <falcon/error.h>
#include <falcon/eng_messages.h>

#include "core_module.h"


namespace Falcon {

namespace core {
class PathObject: public CRObject
{
public:
   PathObject( const CoreClass *genr, Path* path, bool bSerial ):
      CRObject( genr, bSerial )
   {
      if ( path == 0 )
         path = new Path;
         
      setUserData( path );
   }

   PathObject( const PathObject &other );
   virtual ~PathObject();
   virtual PathObject *clone() const;
   Path* getPath() const { return static_cast<Path*>( m_user_data ); }
};


CoreObject* PathObjectFactory( const CoreClass *me, void *path, bool dyn )
{
   return new PathObject( me, static_cast<Path*>( path ), dyn );
}

PathObject::PathObject( const PathObject &other ):
   CRObject( other )
{
   setUserData( new Path( *getPath() ) );
}

PathObject::~PathObject()
{
   delete getPath();
}

PathObject *PathObject::clone() const
{
   return new PathObject( *this );
}

/*# @class Path
   @brief Interface to local filesystem path definition.
   @optparam path The path that will be used as initial path.
   @raise ParamError in case the inital path is malformed.

   This class offers an object oriented interface to access
   path elements given a complete path, or to build a path from its elements.

   Builds the path object, optionally using the given parameter
   as a complete path constructor.

   If the parameter is an array, it must have at least four
   string elements, and it will be used to build the path from
   its constituents. For example:

   @code
      unit = "C"
      location = "/a/path/to"
      file = "somefile"
      ext = "anext"
      p = Path( [ unit, location, file, ext ] )
   @endcode

   @b nil can be passed if some part of the specification is not used.

   The path (or any part of it) may be specified both in RFC3986 format or in
   MS-Windows path format.

   @note Use the fileNameMerge() function to simply merge elements of a path
   specification into a string.

   @see fileNameMerge

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


/*# @property fulloc Path
   @brief Unit specificator and location.
   @raise ParamError if assigned to a value that makes the path invalid.

   This property contains the location of this path, including the unit
   specificator, if present.

   So, in a path like "/C:/path/to/me.txt", the @b fulloc property
   (notice the two 'l' characters in the name) will have the value of
   "/C:/path/to", while in a relative path like "relative/file.txt"
   it will take the same value of @a Path.location.

   Assigning a value to this property means to change the value of
   both the unit specificator and location at the same time.
*/

/*# @property file Path
   @brief File part.
   @raise ParamError if assigned to a value that makes the path invalid.

   This element coresponds to the first part of the file element, if it is
   divided into a filename and an extension by a "." dot.

   @note 
   If an extension is given, then @b filename is the same as @b file + "." + @b extension 
*/

/*# @property filename Path
   @brief File name part.
   @raise ParamError if assigned to a value that makes the path invalid.

   This is the part of the path that identifies an element in a directory.
   It includes everything after the last "/" path separator.
   
   @note 
   If an extension is given, then @b filename is the same as @b file + "." + @b extension 
*/

/*# @property extension Path
   @brief File extension part.
   @raise ParamError if assigned to a value that makes the path invalid.

   This element coresponds to the first last of the file element, if it is
   divided into a filename and an extension by a "." dot.

   @note 
   If an extension is given, then @b filename is the same as @b file + "." + @b extension 
*/

/*# @property path Path
   @brief Complete path.
   @raise ParamError if assigned to a value that makes the path invalid.

   This is the complete path referred by this object.
*/


/*# @property winpath Path
   @brief Complete path in MS-Windows format.

   This is the complete path referred by this object, given in MS-Windows
   format.

   Use this if you need to produce scripts or feed it into external process
   on windows platforms. Normally, all the I/O functions used by Falcon
   on any platform understand the RFC3986 format.

   @note The property is read-only; you can anyhow assign a path in MS-Windows
   format to the @a Path.path property.
*/

/*# @property winloc Path
   @brief Complete path in MS-Windows format.

   This is the location element in the complete path, given in MS-Windows
   format.

   Use this if you need to produce scripts or feed it into external process
   on windows platforms. Normally, all the I/O functions used by Falcon
   on any platform understand the RFC3986 format.

   @note The property is read-only; you can anyhow assign a location in MS-Windows
   format to the @a Path.location property.
*/


/*# @property winfulloc Path
   @brief Complete path in MS-Windows format.

   This is the full location element in this path (unit specificator + location), 
   given in MS-Windows format.

   Use this if you need to produce scripts or feed it into external process
   on windows platforms. Normally, all the I/O functions used by Falcon
   on any platform understand the RFC3986 format.

   @note The property is read-only; you can anyhow assign a full location in MS-Windows
   format to the @a Path.fulloc property.
*/


// Reflective URI method
void Path_path_rfrom(CoreObject *instance, void *user_data, Item &property, const PropEntry& )
{
   Path &path = *static_cast<Path *>( user_data );

   if ( ! path.isValid() )
   {
      VMachine* vm = VMachine::getCurrent();
      throw new ParamError( ErrorParam( e_inv_params ).
         origin( e_orig_runtime ).
         extra( vm != 0 ? vm->moduleString( rtl_invalid_path ) : "" ) );
   }

   // And now set the property
   if ( property.isString() )
      property.asString()->bufferize( path.get() );
   else
      property = new CoreString( path.get() );
}

void Path_path_rto(CoreObject *instance, void *user_data, Item &property, const PropEntry& )
{
   Path &path = *static_cast<Path *>( user_data );

   // We're setting the URI, that is, the property has been written.
   FALCON_REFLECT_STRING_TO( (&path), set );

   if ( ! path.isValid() )
   {
      VMachine* vm = VMachine::getCurrent();
      throw new ParamError( ErrorParam( e_inv_params ).
         origin( e_orig_runtime ).
         extra( vm != 0 ? vm->moduleString( rtl_invalid_path ) : "" ) );
   }
}

// Reflective URI method
void Path_filename_rfrom(CoreObject *instance, void *user_data, Item &property, const PropEntry& )
{
   Path &path = *static_cast<Path *>( user_data );
   FALCON_REFLECT_STRING_FROM( (&path), getFilename );
}

void Path_filename_rto(CoreObject *instance, void *user_data, Item &property, const PropEntry& )
{
   Path &path = *static_cast<Path *>( user_data );

   // We're setting the path, that is, the property has been written.
   FALCON_REFLECT_STRING_TO( (&path), setFilename );

   if ( ! path.isValid() )
   {
      VMachine* vm = VMachine::getCurrent();
      throw new ParamError( ErrorParam( e_inv_params ).
         origin( e_orig_runtime ).
         extra( vm != 0 ? vm->moduleString( rtl_invalid_path ) : "" ) );
   }
}

// Reflective path method
void Path_file_rfrom(CoreObject *instance, void *user_data, Item &property, const PropEntry& )
{
   Path &path = *static_cast<Path *>( user_data );
   FALCON_REFLECT_STRING_FROM( (&path), getFile );
}

void Path_file_rto(CoreObject *instance, void *user_data, Item &property, const PropEntry& )
{
   Path &path = *static_cast<Path *>( user_data );
   // We're setting the path, that is, the property has been written.
   FALCON_REFLECT_STRING_TO( (&path), setFile );

   if ( ! path.isValid() )
   {
      VMachine* vm = VMachine::getCurrent();
      throw new ParamError( ErrorParam( e_inv_params ).
         origin( e_orig_runtime ).
         extra(  vm != 0 ? vm->moduleString( rtl_invalid_path ) : "" ) );
   }
}

// Reflective path method
void Path_location_rfrom(CoreObject *instance, void *user_data, Item &property, const PropEntry& )
{
   Path &path = *static_cast<Path *>( user_data );
   FALCON_REFLECT_STRING_FROM( (&path), getLocation );
}

void Path_location_rto(CoreObject *instance, void *user_data, Item &property, const PropEntry& )
{
   Path &path = *static_cast<Path *>( user_data );
   // We're setting the path, that is, the property has been written.
   FALCON_REFLECT_STRING_TO( (&path), setLocation );

   if ( ! path.isValid() )
   {
      VMachine* vm = VMachine::getCurrent();
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ ).
         origin( e_orig_runtime ).
         extra( vm != 0 ? vm->moduleString( rtl_invalid_path ) : "" ) );
   }
}

// Reflective full location method
void Path_fullloc_rfrom(CoreObject *instance, void *user_data, Item &property, const PropEntry& )
{
   Path &path = *static_cast<Path *>( user_data );
   FALCON_REFLECT_STRING_FROM( (&path), getFullLocation );
}

void Path_fullloc_rto(CoreObject *instance, void *user_data, Item &property, const PropEntry& )
{
   Path &path = *static_cast<Path *>( user_data );
   // We're setting the path, that is, the property has been written.
   FALCON_REFLECT_STRING_TO( (&path), setFullLocation );

   if ( ! path.isValid() )
   {
      VMachine* vm = VMachine::getCurrent();
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ ).
         origin( e_orig_runtime ).
         extra( vm != 0 ? vm->moduleString( rtl_invalid_path ) : "" ) );
   }
}

// Reflective path method
void Path_unit_rfrom(CoreObject *instance, void *user_data, Item &property, const PropEntry& )
{
   Path &path = *static_cast<Path *>( user_data );
   FALCON_REFLECT_STRING_FROM( (&path), getResource );
}

void Path_unit_rto(CoreObject *instance, void *user_data, Item &property, const PropEntry& )
{
   Path &path = *static_cast<Path *>( user_data );
   // We're setting the path, that is, the property has been written.
   FALCON_REFLECT_STRING_TO( (&path), setResource );

   if ( ! path.isValid() )
   {
      VMachine* vm = VMachine::getCurrent();
      throw new ParamError( ErrorParam( e_inv_params ).
         origin( e_orig_runtime ).
         extra( vm != 0 ? vm->moduleString( rtl_invalid_path ) : "" ) );
   }
}


// Reflective path method
void Path_extension_rfrom(CoreObject *instance, void *user_data, Item &property, const PropEntry& )
{
   Path &path = *static_cast<Path *>( user_data );
   FALCON_REFLECT_STRING_FROM( (&path), getExtension );
}

void Path_extension_rto(CoreObject *instance, void *user_data, Item &property, const PropEntry& )
{
   Path &path = *static_cast<Path *>( user_data );
   // We're setting the path, that is, the property has been written.
   FALCON_REFLECT_STRING_TO( (&path), setExtension );

   if ( ! path.isValid() )
   {
      VMachine* vm = VMachine::getCurrent();
      throw new ParamError( ErrorParam( e_inv_params ).
         origin( e_orig_runtime ).
         extra( vm != 0 ? vm->moduleString( rtl_invalid_path ) : "" ) );
   }
}



// Reflective windows method
void Path_winpath_rfrom(CoreObject *instance, void *user_data, Item &property, const PropEntry& )
{
   Path &path = *static_cast<Path *>( user_data );
   FALCON_REFLECT_STRING_FROM( (&path), getWinFormat );
}

// Reflective windows method
void Path_winloc_rfrom(CoreObject *instance, void *user_data, Item &property, const PropEntry& )
{
   Path &path = *static_cast<Path *>( user_data );
   FALCON_REFLECT_STRING_FROM( (&path), getWinLocation );
}

// Reflective windows method
void Path_winfulloc_rfrom(CoreObject *instance, void *user_data, Item &property, const PropEntry& )
{
   Path &path = *static_cast<Path *>( user_data );
   FALCON_REFLECT_STRING_FROM( (&path), getFullWinLocation );
}



FALCON_FUNC  Path_init ( ::Falcon::VMachine *vm )
{
   Item *p0 = vm->param(0);
   // no parameter? -- ok, we have an empty path
   if ( p0 == 0 )
      return;

   // extract the path instance created by the factory function      
   
   if ( ( ! p0->isString() && ! p0->isArray() ) )
   {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ ).
         origin( e_orig_runtime ).extra( "[S|A]" ) );
   }

   PathObject *self = dyncast<PathObject*>( vm->self().asObject() );
   Path *path = self->getPath();
   
   if ( p0->isString() )
   {
      path->set( *p0->asString() );
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

      path->join( *unitspec, *fpath, *fname, *fext );
   }

   if ( ! path->isValid() )
   {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ ).
         origin( e_orig_runtime ).
         extra( vm->moduleString( rtl_invalid_path ) ) );
   }
}

}
}

/* end of path_ext.cpp */
