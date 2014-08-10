/*
   FALCON - The Falcon Programming Language.
   FILE: genericdata.h

   Generic item with standardized behavior.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Fri, 12 Aug 2011 21:06:49 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/


#ifndef _FALCON_GENERICDATA_H_
#define _FALCON_GENERICDATA_H_

#include <falcon/setup.h>
#include <falcon/string.h>

namespace Falcon
{

/** Generic data for items with standardized behavior.
 
 This simple class offers a base for items shared with the engine through 
 ClassGeneric. This is meant to be subclasses by instances with minimal 
 requirements. See ClassGeneric for more details.
 */
class FALCON_DYN_CLASS GenericData
{
public:
   GenericData( const String& name ):
      m_name( name )
   {}
   
   const String& name() const { return m_name; }
      
   virtual ~GenericData() {}
   virtual void gcMark( uint32 value ) = 0;
   virtual bool gcCheck( uint32 value ) = 0;
   
   /** Return a copy of this generic item, or 0 if not allowed to copy. */
   virtual GenericData* clone() const = 0;
   virtual void describe( String& target ) const = 0;
   
   /** Returns an instance of ClassGeneric */
   static const Class* handler();

private:
   String m_name;
};

}

#endif	/* _FALCON_GENERICDATA_H_ */

/* end of genericdata.h */
