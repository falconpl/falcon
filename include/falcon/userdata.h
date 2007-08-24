/*
   FALCON - The Falcon Programming Language.
   FILE: userdata.h
   $Id: userdata.h,v 1.4 2007/07/21 11:59:12 jonnymind Exp $

   Embeddable falcon object user data.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: sab giu 23 2007
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
   Embeddable falcon object user data.
*/

#ifndef flc_userdata_H
#define flc_userdata_H

#include <falcon/setup.h>
#include <falcon/basealloc.h>

namespace Falcon {

class Item;
class String;

/** Embeddable falcon object user data.
   An instance of this class can be set as "user data" of Falcon scripts objects.

   This class defines minimal a interface that is used by Falcon objects to
   automatically synchronize the status of the objects and the status of UserData
   instances.

   Most of the times, this relflection is not needed: when the Falcon objects are mainly
   manipulated by scripts, the application will just read the values in the properties,
   and set them properly to update system status; conversely, when the application uses
   the carried object heavily, and the scripts need just to check some of their values
   from now and then, the module interfaces can just provide some accessors to read the
   desired data.

   Also, being Falcon objects much more flexible than C++ objects, the modules and
   applications are often willing to provide Falcon objects with more featrures and
   abilities than the ones merely provided by the user data. It's the case of the
   Falcon RTL Stream class, which uses a C++ Falcon::Stream as core data but relies on
   that only for some operations. The vast majority of features are directly provided
   to the scripts by the module, and the internal instance of Falcon::Stream is used
   merely as an utility to access the file system.

   Nevertheless, reflection comes handy when the load balance of object usage between
   application and scripts is even, and when the application C++ object, or a subset
   of its interface, is meaningfull to the scripts as-is.

   Subclasses willing to use reflection should overload isReflective() method so that
   it returns true. In this case, CoreObject instances shared by the scripts will
   call the getProperty() and setProperty() methods of UserData to update their internal
   property table each time an access to a certain member is performed.

   The reflection model doesn't need to be complete: the
   Falcon object may provide properties that the the UserData subclass is not willing
   to reflect, and the UserData subclasses may have properties and methods that won't
   be visible by scripts. If a UserData subclass is not willing to manage a property,
   it may just return ignore it in the handler method.

   Applications may not be willing to derive their internal data from UserData to share
   it with a script. In that case, they can provide a "carrier", which is just a subclass
   of UserData owning a reference of the internal application data (or just pointing at
   it, if it's known that the application data will survive script execution time).
*/

class FALCON_DYN_CLASS UserData: public BaseAlloc
{
public:
   UserData() {}
   virtual ~UserData();

   /** Declare if a certain subclass of UserData is reflective. */
   virtual bool isReflective();

   /** Get given property.
      When this method is called, the reflected object has already determined that
      it can access the property.

      \param propName the name of the property to be read.
      \param prop the item that should assume the value of the property
   */
   virtual void getProperty( const String &propName, Item &prop );

   /** Set given property.
      When this method is called, the reflected object has already changed the
      property in the item.

      \param propName the name of the property to be read.
      \param prop the item that should assume the value of the property
   */
   virtual void setProperty( const String &propName, Item &prop );

   virtual UserData *clone();
};

}

#endif

/* end of userdata.h */
