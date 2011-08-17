/*
   FALCON - The Falcon Programming Language.
   FILE: vardef.h

   Special compile-time definition for variables
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sat, 11 Jul 2009 20:42:43 +0200

   -------------------------------------------------------------------
   (C) Copyright 2009: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef FLC_VARDEF_H
#define FLC_VARDEF_H

#include <falcon/setup.h>
#include <falcon/types.h>
#include <falcon/common.h>
#include <falcon/reflectfunc.h>

namespace Falcon {

class Symbol;
class Stream;

/** Variable initial value definition.

   This class holds the immediate values of properties of classes and objects,
   and eventually the values static symbols, when an initial value has been declared
   if it's declared. If the properties are declared as expressions, or anyhow
   not as immediate values, the relative property definition is set to nil and
   the code generator will create an internal constructor (called before the
   user-defined constructor) that will initialize the complex property values.

   Property definition can then be:
   - nil (actually declared nil or to be filled by the internal construtor)
   - integer
   - numeric
   - string (only the string ID in the module string table is stored)
   - symbol (only the symbol ID in the module symbol table is stored)
*/
class FALCON_DYN_CLASS VarDef: public BaseAlloc
{
public:
   typedef enum {
      t_nil,
      t_int,
      t_bool,
      t_num,
      t_string,
      t_symbol,
      t_base,
      t_reference,
      t_reflective,
      t_reflectFunc
   } t_type;
private:
   t_type m_val_type;
   bool m_bReadOnly;

   union {
      bool val_bool;
      uint64 val_int;
      numeric val_num;

      struct {
         t_reflection mode;
         uint32 offset;
      } val_reflect;

      struct {
         reflectionFunc from;
         reflectionFunc to;
         void *data;
      } val_rfunc;

      const String *val_str;
      Symbol *val_sym;
   } m_value;

public:

   VarDef():
      m_val_type(t_nil),
      m_bReadOnly( false )
   {}

   explicit VarDef( bool val ):
      m_val_type(t_bool),
      m_bReadOnly( false )
   {
      m_value.val_bool = val;
   }

   VarDef( int64 val ):
      m_val_type(t_int),
      m_bReadOnly( false )
   {
      m_value.val_int = val;
   }

   VarDef( numeric val ):
      m_val_type(t_num),
      m_bReadOnly( false )
   {
      m_value.val_num = val;
   }

   VarDef( const String *str ):
      m_val_type(t_string),
      m_bReadOnly( false )
   {
      m_value.val_str = str;
   }

   VarDef( Symbol *sym ):
      m_val_type( t_symbol ),
      m_bReadOnly( false )
   {
      m_value.val_sym = sym;
   }

   VarDef( t_type t, Symbol *sym ):
      m_val_type( t ),
      m_bReadOnly( false )
   {
      m_value.val_sym = sym;
   }

   VarDef( t_type t, int64 iv ):
      m_val_type(t),
      m_bReadOnly( false )
   {
      m_value.val_int = iv;
   }

   VarDef( reflectionFunc rfrom, reflectionFunc rto=0 ):
      m_val_type( t_reflectFunc ),
      m_bReadOnly( rto==0 )
   {
      m_value.val_rfunc.from = rfrom;
      m_value.val_rfunc.to = rto;
   }

   VarDef( t_reflection mode, uint32 offset ):
      m_val_type( t_reflective ),
      m_bReadOnly( false )
   {
      m_value.val_reflect.mode = mode;
      m_value.val_reflect.offset = offset;
   }

   t_type type() const { return m_val_type; }

   /** Describes this property as nil.
      \return a reference to this instance, for variable parameter initialization idiom.
   */
   VarDef& setNil() { m_val_type = t_nil; return *this; }
   VarDef& setBool( bool val ) { m_val_type = t_bool; m_value.val_bool = val; return *this;}
   VarDef& setInteger( int64 val ) { m_val_type = t_int; m_value.val_int = val; return *this;}
   VarDef& setString( const String *str ) { m_val_type = t_string; m_value.val_str = str; return *this; }
   VarDef& setSymbol( Symbol *sym ) { m_val_type = t_symbol; m_value.val_sym = sym; return *this;}
   VarDef& setNumeric( numeric val ) { m_val_type = t_num; m_value.val_num = val; return *this;}
   VarDef& setBaseClass( Symbol *sym ) { m_val_type = t_base; m_value.val_sym = sym; return *this;}
   VarDef& setReference( Symbol *sym ) { m_val_type = t_reference; m_value.val_sym = sym; return *this;}

   /** Describes this property as reflective.
      This ValDef defines a property that will have user functions called when the VM wants
      to set or get a property.

      It is also possible to define an extra reflective data, that should be alive during the
      lifespan of the module defining it, that will be passed back to the property set/get callback
      functions as the \a PropEntry::reflect_data property of the "entry" parameter.

      \param rfrom Function that gets called when the property is \b set \b from an external source.
      \param rto Function that gets called when the property is \b read and then stored \b to the external
         source; set to 0 to have a read-only reflective property.
      \param reflect_data a pointer that will be passed as a part of the entry structure in the callback
             method.
      \return a reference to this instance, for variable parameter initialization idiom.
   */
   VarDef &setReflectFunc( reflectionFunc rfrom, reflectionFunc rto=0, void *reflect_data = 0 ) {
      m_val_type = t_reflectFunc;
      m_bReadOnly = rto == 0;
      m_value.val_rfunc.from = rfrom;
      m_value.val_rfunc.to = rto;
      m_value.val_rfunc.data = reflect_data;
      return *this;
   }

   /** Describes this property as reflective.
      \return a reference to this instance, for variable parameter initialization idiom.
   */
   VarDef &setReflective( t_reflection mode, uint32 offset )
   {
      m_val_type = t_reflective;
      m_value.val_reflect.mode = mode;
      m_value.val_reflect.offset = offset;
      return *this;
   }

   /** Describes this property as reflective.
      Shortcut calculating the offset given a sample structure and a field in that.
      \return a reference to this instance, for variable parameter initialization idiom.
   */
   VarDef& setReflective( t_reflection mode, void *base, void *position )
   {
      return setReflective( mode, static_cast<uint32>(
         static_cast<char *>(position) - static_cast<char *>(base)) );
   }

   /** Describes this property as reflective.
      \return a reference to this instance, for variable parameter initialization idiom.
   */
   VarDef& setReadOnly( bool ro ) {
      m_bReadOnly = ro;
      return *this;
   }


   bool asBool() const { return m_value.val_bool; }
   int64 asInteger() const { return m_value.val_int; }
   const String *asString() const { return m_value.val_str; }
   Symbol *asSymbol() const { return m_value.val_sym; }
   numeric asNumeric() const { return m_value.val_num; }
   reflectionFunc asReflectFuncFrom() const { return m_value.val_rfunc.from; }
   reflectionFunc asReflectFuncTo() const { return m_value.val_rfunc.to; }
   void* asReflectFuncData() const { return m_value.val_rfunc.data; }
   t_reflection asReflecMode() const { return m_value.val_reflect.mode; }
   uint32 asReflecOffset() const { return m_value.val_reflect.offset; }

   bool isNil() const { return m_val_type == t_nil; }
   bool isBool() const { return m_val_type == t_bool; }
   bool isInteger() const { return m_val_type == t_int; }
   bool isString() const { return m_val_type == t_string; }
   bool isNumeric() const { return m_val_type == t_num; }
   bool isSymbol() const { return m_val_type == t_symbol || m_val_type == t_base; }
   bool isBaseClass() const { return m_val_type == t_base; }
   bool isReference() const { return m_val_type == t_reference; }
   bool isReflective() const { return m_val_type == t_reflective; }
   bool isReflectFunc() const { return m_val_type == t_reflectFunc; }
   bool isReadOnly() const { return m_bReadOnly; }

   bool save( const Module *mod, Stream *out ) const;
   bool load( const Module *mod, Stream *in );
};

}

#endif

/* end of vardef.h */
