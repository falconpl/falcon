/*
   FALCON - The Falcon Programming Language.
   FILE: citerator.h
   $Id: citerator.h,v 1.3 2007/06/30 10:58:07 jonnymind Exp $

   Base abstract class for generic collection iterators.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: dom giu 24 2007
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
   Base abstract class for generic collection iterators.
*/

#ifndef flc_citerator_H
#define flc_citerator_H

#include <falcon/setup.h>
#include <falcon/types.h>
#include <falcon/userdata.h>

namespace Falcon {

/**
   Base abstract class for generic collection iterators.
   This is also used as internal object for iterators.
*/
class FALCON_DYN_CLASS CoreIterator: public UserData
{
protected:
   CoreIterator() {}

public:

   virtual bool next() = 0;
   virtual bool prev() = 0;
   virtual bool hasNext() const = 0;
   virtual bool hasPrev() const = 0;
   /** Must be called after an isValid() check */
   virtual Item &getCurrent() const = 0;

   virtual bool isValid() const = 0;
   virtual bool isOwner( void *collection ) const = 0;
   virtual bool equal( const CoreIterator &other ) const = 0;
   virtual bool erase() = 0;
   virtual bool insert( const Item &item ) = 0;



   virtual void invalidate() = 0;


};

}

#endif

/* end of citerator.h */
