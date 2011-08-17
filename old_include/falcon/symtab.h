/*
   FALCON - The Falcon Programming Language.
   FILE: flc_symtab.h

   Symbol table definition
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: mar ago 3 2004

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef FLC_SYMBOLTABLE_H
#define FLC_SYMBOLTABLE_H

#include <falcon/setup.h>
#include <falcon/common.h>
#include <falcon/string.h>
#include <falcon/genericvector.h>
#include <falcon/genericmap.h>
#include <falcon/basealloc.h>

namespace Falcon {

class Symbol;
class Stream;
class Module;

class FALCON_DYN_CLASS SymbolTable: public BaseAlloc
{
   /** Internal symbol map.
      (const String *, Symbol * )
   */
   Map m_map;

public:
   /** Constructs the symbol table.
      The symbol table is built with owndership of the symbols.
   */
   SymbolTable();

   /** Adds a symbol.
      The name of the symbol is used as the key in the symbol table.
      \param sym the symbol to be added.
      \return true if the symbol can be added, false if it were already present.
   */
   bool add( Symbol *sym );

   /** Seek a symbol given its name.
      If the symbol with the given name can't be found in the symbol
      table, the function returns null.
      \note it is possible also to call this function with static C strings from
            code.
      \param name the name of the symbol to be found.
      \return a pointer to the symbol if it's found, or 0 otherwise.
   */
   Symbol *findByName( const String &name ) const
   {
      Symbol **ptrsym = (Symbol **) m_map.find( &name );
      if ( ptrsym == 0 )
         return 0;
      return *ptrsym;
   }

   /** Exports all the undefined symbols.
      Used by the compiler if the module being compiled asked for complete export.
   */
   void exportUndefined();

   /** Remove a symbol given it's name.
      If the SymbolTable has the ownership of the symbols, then the symbol is
      destroyed. The symbol name is never destroyed though.

      \param name the name of the symbol to be destroyed.
      \return true if the name can be found, false otherwise.
   */
   bool remove( const String &name );


   /** Return the number of symbols stored in this table.
      \return the number of symbols stored in this table.
   */
   int size() const {
      return m_map.size();
   }


   /** Save the symbol table on a stream.
      The serialization of the table involves only saving the ID of the strings
      and of the symbols that are in the table.

      \param out the stream where the table must be saved.
      \return true if the operation has success, false otherwise.
   */
   bool save( Stream *out ) const;

   /** Load the symbol table from a stream.
      If the symbol table owns the symbols, then the symbols are serialized on the
      stream too; otherwise, only the ID are saved.

      The deserialization may return false if the function detects some problem,
      i.e. an invald format.
      \param owner the module for which the symbols are created.
      \param in the input stream where the table must be loaded from.
      \return true if the operation has success, false otherwise.
   */
   bool load( const Module *owner, Stream *in );

   const Map &map() const { return m_map; }
};

class FALCON_DYN_CLASS SymbolVector: public GenericVector
{

public:
   SymbolVector();

   ~SymbolVector();

   Symbol *symbolAt( uint32 pos ) const { return *(Symbol **) at( pos ); }

   bool save( Stream *out ) const;
   bool load( Module *owner, Stream *in );
};

}
#endif
/* end of flc_symtab.h */
