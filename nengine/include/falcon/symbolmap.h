/*
   FALCON - The Falcon Programming Language.
   FILE: symbolmap.h

   A simple class orderly guarding symbols and the module they come from.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Tue, 09 Aug 2011 00:43:47 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef _FALCON_SYMBOLMAP_H_
#define _FALCON_SYMBOLMAP_H_

#include <falcon/setup.h>
#include <falcon/string.h>

namespace Falcon 
{

class Module;
class Symbol;

/** A simple class orderly guarding symbols and the module they come from.
 */
class FALCON_DYN_CLASS SymbolMap
{
public:

   /** An entry of a the symbol map.
    An entry in a module map is composed of:
    - A pointer to the symbol;
    - The pointer to the module where the symbol comes from.
    */
      
   class Entry
   {
   public:
      
      Entry( Symbol* sym, Module* mod ):
         m_symbol(sym),
         m_module(mod)
      {}
      
      ~Entry() {};

      Symbol* symbol() const { return m_symbol; }
      Module* module() const { return m_module; }

   private:
      Symbol* m_symbol;
      Module* m_module;
   };
  
   SymbolMap();
   ~SymbolMap();
   
   void add( Symbol* sym, Module* mod );
   void remove( const String& symName );
   Entry* find( const String& symName ) const;
   

   class EntryEnumerator 
   {
   public:
      virtual ~EntryEnumerator()=0;
      virtual void operator()( Entry* e );
   };
   
   void enumerate( EntryEnumerator& rator ) const;
   
private:
   class Private;
   Private* _p;
};

}

#endif	/* _FALCON_SYMBOLMAP_H_ */

/* end of symbolmap.h */
