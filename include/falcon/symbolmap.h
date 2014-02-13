/*
   FALCON - The Falcon Programming Language.
   FILE: symbolmap.h

   Map holding local and global variable tables.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sat, 29 Dec 2012 10:03:38 +0100

   -------------------------------------------------------------------
   (C) Copyright 2012: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef _FALCON_SYMBOLMAP_H_
#define _FALCON_SYMBOLMAP_H_

#include <falcon/setup.h>
#include <falcon/types.h>

namespace Falcon {

class String;
class Variable;
class Symbol;

/** Map holding local and global variable tables.
 *
 * This map holds variables that can be defined in functions
 * and modules.
 *
 * It has support to find a variable name given its type and
 * ID.
 *
 */
class FALCON_DYN_CLASS SymbolMap
{
public:
   SymbolMap();
   SymbolMap( const SymbolMap& other );
   ~SymbolMap();

   /** Adds a parameter to this function.
    \param name The parameter to be added.
    \return the parameter ID assigned to this variable.
    */
   int32 insert( const String& name );

   int32 insert( const Symbol* sym );

   int32 find( const String& name ) const;
   int32 find( const Symbol* sym ) const;

   const String& getNameById( uint32 id ) const;
   const Symbol* getById( uint32 id ) const;

   uint32 size() const;

   class Enumerator {
   public:
      virtual void operator()( const String& name ) = 0;
   };

   void enumerate( Enumerator& e );

   void store( DataWriter* dw ) const;
   void restore( DataReader* dr );

private:
   class Private;
   SymbolMap::Private* _p;
};

}

#endif

/* end of symbolmap.h */
