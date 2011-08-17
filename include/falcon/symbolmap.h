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
class ModGroup;

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
   
   /** Adds a symbol that is globally exported.
    \param sym The symbol to be exported.
    \param mod Optinally, the module that is exporting the symbol.
    \return true if the symbol could be added, false if it was already present.
    */
   
   bool add( Symbol* sym, Module* mod = 0 );
   
   /** Removes an exported symbol.
    \param symName The name of the symbol to be exported.
    */
   void remove( const String& symName );
   
   /** Finds a symbol that is globally exported or globally defined.
    \param name The name of the symbol that is exported.    
    \return An entry containing the symbol and the declarer, if defined, 
            or 0 if the name cannot be found.
    
    \note The caller must be prepared to the event that a symbol is found, but
    the \b declarer parameter is set to zero. In fact, it is possible for embedding
    applications to create module-less symbols.
    */

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
   
   friend class ModGroup;
};

}

#endif	/* _FALCON_SYMBOLMAP_H_ */

/* end of symbolmap.h */
