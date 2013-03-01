/*
   FALCON - The Falcon Programming Language.
   FILE: streamtraits.h

   General traits on which streams are based.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Thu, 28 Feb 2013 18:18:29 +0100

   -------------------------------------------------------------------
   (C) Copyright 2013: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef _FALCON_STREAMTRAITS_H_
#define _FALCON_STREAMTRAITS_H_

#include <falcon/setup.h>
#include <falcon/types.h>
#include <falcon/string.h>

namespace Falcon
{

class Selector;
class Multiplex;

/**
 * General stream traits.
 *
 * This class is used to describe a family of streams
 * being all alike.
 */
class FALCON_DYN_CLASS StreamTraits
{
public:
   /**
    * \param name A symbolic name for this kind of streams.
    * \param module A module where this multiplex code is stored.
    *
    * \note  The module is used for back-reference and keep alive marks.
    * Usually, the value is implicit, in the sense that the code is in a module
    * it knows, so it's not necessary to get this
    */
   StreamTraits( const String& name, Module* mod = 0 ):
      m_module( mod ),
      m_name( name )
   {}

   virtual ~StreamTraits() {}

   const String& name() const { return m_name; }

   /**
    * Creates a concrete instance of a multiplex handling streams of this type.
    * \return new multiplex instance.
    * \param master The owner of this new multiplex.

    */
   virtual Multiplex* multiplex( Selector* master ) const = 0;

protected:
   Module* m_module;

private:
   String m_name;
};

}

#endif

/* end of streamtraits.h */
