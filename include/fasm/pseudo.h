/*
   FALCON - The Falcon Programming Language.
   FILE: fasm_pseudo.h

   Pseudo operators
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: dom giu 20 2004

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef FASM_PSEUDO_H
#define FASM_PSEUDO_H

#include <falcon/string.h>
#include <falcon/uintpair.h>
#include <falcon/genericlist.h>
#include <falcon/traits.h>
#include <falcon/basealloc.h>
#include <fasm/labeldef.h>

namespace Falcon {

class Symbol;
class Stream;

/** Pseudo is the minimal semantic unit for a HASM file.
   This is very alike what is SyntreeElement for Falcon, but much more simpler.
   A pseudo, short for "pseudo operation or assembly instruction", can be any of the
   following:
      - Line declaration for debugging purposes
      - Directive (.global, .function and so on )
      - Assembly Instruction
      - Label

   Actually, as instructions and directives are parsed as they are scanned, there's no
   need to record them, so Pseudo does only the job to record instruction names and
   poperators while the lexer scans for them.

*/

class FALCON_DYN_CLASS Pseudo: public BaseAlloc
{

public:

   typedef enum {
      tnil = 0,
      imm_true,
      imm_false,
      imm_int,
      imm_double,
      imm_string,
      imm_range,
      tdispstring,
      tlbind,
      tname,
      tsymbol,
      tswitch_list,
      tregA,
      tregB,
      tregS1,
      tregS2,
      tregL1,
      tregL2
   } type_t;


private:
   typedef union {
      int64 intval;
      uint32 posval;
      numeric dblval;
      LabelDef *ldval;
      String *strval;
      String *dispstring;
      Symbol *symval;
      List *child;
      struct {
         int32 start;
         int32 end;
      } range;
   } pseudo_value;

   type_t m_type;
   int m_line;

   pseudo_value m_value;
   bool m_disp;
   bool m_fixed;

public:

   Pseudo( type_t opt, bool disp=true ):
      m_type( opt ),
      m_disp(disp),
      m_fixed( false )
   {
      if ( opt == tswitch_list )
         m_value.child = new List;
   }

   explicit Pseudo( int l, type_t opt, String *str, bool disp=true  ):
      m_line( l ),
      m_type( opt ),
      m_disp( disp ),
      m_fixed( false )
   {
      m_value.strval = str;
   }

   Pseudo( int l, type_t opt, const char *str, bool disp=true  );

   Pseudo( int l, int64 intval, bool disp=true ):
      m_line( l ),
      m_type( imm_int ),
      m_disp( disp ),
      m_fixed( false )
   {
      m_value.intval = intval;
   }

   explicit Pseudo( int l, type_t opt, int64 intval, bool disp=true  ):
      m_line( l ),
      m_type( opt ),
      m_disp( disp ),
      m_fixed( false )
   {
      m_value.intval = intval;
   }

   Pseudo( int l, double dval, bool disp=true  ):
      m_line( l ),
      m_type( imm_double ),
      m_disp( disp ),
      m_fixed( false )
   {
      m_value.dblval = dval;
   }

   Pseudo( int l, type_t t, Symbol *sym, bool disp=true  ):
      m_line( l ),
      m_type( t ),
      m_disp( disp ),
      m_fixed( false )
   {
      m_value.symval = sym;
   }

   Pseudo( int l, LabelDef *def, bool disp=true  ):
      m_line( l ),
      m_type( tname ),
      m_disp( disp ),
      m_fixed( false )
   {
      m_value.ldval = def;
   }

   explicit Pseudo( int l, int32 start, int32 end, bool disp=true  ):
      m_line( l ),
      m_type( imm_range ),
      m_disp( disp ),
      m_fixed( false )
   {
      m_value.range.start = start;
      m_value.range.end = end;
   }

   ~Pseudo();

   type_t type() const { return m_type; }
   void type( const type_t t ) { m_type = t; }

   int line() const { return m_line; }
   void line( const int l ) { m_line = l; }

   int64 asInt() const { return m_value.intval; }
   numeric asDouble() const { return m_value.dblval; }
   uint32 asPosition() const { return m_value.posval; }
   const String &asString() const { return *m_value.strval; }
   String &asString() { return *m_value.strval; }
   Symbol *asSymbol() const { return m_value.symval; }
   LabelDef *asLabel() const { return m_value.ldval; }
   List *asList() { return m_value.child; }
   bool disposeable() const { return m_disp; }
   void disposeable( bool d ) { m_disp = d; }
   int32 asRangeStart() const { return m_value.range.start; }
   int32 asRangeEnd() const { return m_value.range.end; }

   /** Non variable pseudocode.
      This instructs the assembler what kind of parameter to generate:
      variable accordingly with the type of the item or "fixed".

      In case of fixed parameters, the VM won't decode them and will
      leave the decoding to the instruction handlers.

      The compiler will generate the parameter specificator 0E or 0F
      (fixed 32 or 64 bit parameter) accordingly to the type of this
      pseudo object.

      A pseudo holding a symbol can be given the fixed status if the
      symbol is to be referred with it's module ID instead of its item ID.
      This happens in the switch block, where the module ID of the
      symbol must be used as there no space (nor need) to store the
      itemid/symbol table indicator couple.

      \param f true to make this pseudo to generate a fixed parameter.
   */
   void fixed( bool f ) { m_fixed = f; }
   bool fixed() const { return m_fixed; }

   void write( Stream *out ) const;

   bool operator <( const Pseudo &other ) const;
};


class PseudoPtrCmp
{
public:
   bool operator() ( const Pseudo *s1, const Pseudo *s2 ) const
      { return  *s1 < *s2; }
};

class PseudoPtrTraits: public VoidpTraits
{
public:
   virtual int compare( const void *first, const void *second ) const;
   virtual void destroy( void *item ) const;
   virtual bool owning() const;
};

namespace traits {
   extern PseudoPtrTraits t_pseudoptr;
}

}

#endif
/* end of fasm_pseudo.h */
