/*
   FALCON - The Falcon Programming Language.
   FILE: flc_symbol.h
   $Id: symbol.h,v 1.10 2007/08/19 09:46:44 jonnymind Exp $

   Short description
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: mar ago 3 2004
   Last modified because:

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
   In order to use this file in its compiled form, this source or
   part of it you have to read, understand and accept the conditions
   that are stated in the LICENSE file that comes boundled with this
   package.
*/

#ifndef FLC_SYMBOL_H
#define FLC_SYMBOL_H

#include <falcon/types.h>
#include <falcon/symtab.h>
#include <falcon/symlist.h>
#include <falcon/genericmap.h>
#include <falcon/basealloc.h>

namespace Falcon {

class Symbol;
class Stream;

/** Variable initiqal value definition definition.
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
      t_reflective
   } t_type;
private:
   t_type m_val_type;

   struct t_reflect {
      int16 offset;
      int16 size;
      bool isSigned;
   };

   union {
      bool val_bool;
      uint64 val_int;
      struct t_reflect val_reflect;
      numeric val_num;
      const String *val_str;
      const Symbol *val_sym;
   } m_value;

public:

   VarDef():
      m_val_type(t_nil)
   {}

   explicit VarDef( bool val ):
      m_val_type(t_bool)
   {
      m_value.val_bool = val;
   }

   VarDef( int64 val ):
      m_val_type(t_int)
   {
      m_value.val_int = val;
   }

   VarDef( numeric val ):
      m_val_type(t_num)
   {
      m_value.val_num = val;
   }

   VarDef( const String *str ):
      m_val_type(t_string)
   {
      m_value.val_str = str;
   }

   VarDef( const Symbol *sym ):
      m_val_type( t_symbol )
   {
      m_value.val_sym = sym;
   }

   VarDef( t_type t, const Symbol *sym ):
      m_val_type( t )
   {
      m_value.val_sym = sym;
   }

   VarDef( t_type t, int64 iv ):
      m_val_type(t)
   {
      m_value.val_int = iv;
   }

   t_type type() const { return m_val_type; }
   void setNil() { m_val_type = t_nil; }
   void setBool( bool val ) { m_val_type = t_bool; m_value.val_bool = val; }
   void setInteger( uint64 val ) { m_val_type = t_int; m_value.val_int = val; }
   void setString( const String *str ) { m_val_type = t_string; m_value.val_str = str; }
   void setSymbol( const Symbol *sym ) { m_val_type = t_symbol; m_value.val_sym = sym; }
   void setNumeric( numeric val ) { m_val_type = t_num; m_value.val_num = val; }
   void setBaseClass( const Symbol *sym ) { m_val_type = t_base; m_value.val_sym = sym; }
   void setReference( const Symbol *sym ) { m_val_type = t_reference; m_value.val_sym = sym; }

   /** Declares this vardef to be reflective.
      This means this vardef is relative to a C or C++ structure that
      can be directly filled, or that can fill directly, data from
      and to CoreObjects instances.

      VarDefs may represent class properties. When the class defined
      by the vardefs is istantiated, the reflectivity information
      passes to the properties in the final core object, and they
      can be used to transfer automatically data to and from
      C structures through CoreObject::configureFrom() and
      CoreObject::configureTo().

      Example:

      \code
         Symbol *someClass = self->addClass( "SomeClass" );
         someClass->setWKS( true );
         some_c_structure cstruct;

         self->addClassProperty( someClass, "field1" )->
            setReflective( (uint16)(&cstruct.field1 - &cstruct), // offset of the field
            sizeof( cstruct.field1 ), // size of the field; 1,2,4 or 8 bytes
            true ); //signed or unsigned data.

         // ... later on, in the program...
         // when the application wants to pass the data to a falcon program:

         CoreObject *BuildFalconObjectFromStruct( VMachine *vm, const some_c_structure &cstruct )
         {
            CoreClass *cls = vm->findWKI( "SomeClass" );
            CoreObject *myobject = cls->createInstance();
            myobject->configureFrom( &cstruct );
            return myobject;
         }

         // And when we want to read back the data...

         void ReadBackData( some_c_structure &cstruct, CoreObject *co );
         {
            co->configureTo( &cstruct );
         }
      \endcode

      \param offset distance from base data of the reflective data
      \param size size of reflective binary data
      \param is True is signed, false if signed.
   */
   void setReflective( uint16 offset, uint16 size, bool is = false ) {
      m_val_type = t_reflective;
      m_value.val_reflect.offset = offset;
      m_value.val_reflect.size = size;
      m_value.val_reflect.isSigned = is;
   }

   /** Declares this vardef to be reflective.
      This is just a shortcut that calculates the offset distance between
      the data (base address) and the dest (field address).

      So, it actually calls setReflective( uint16, uint16, bool ),
      passing (dest - data) as offset.
      Example:
      \code
         Symbol *someClass = self->addClass( "SomeClass" );

         some_c_structure cstruct;

         self->addClassProperty( someClass, "field1" )->
            setReflective( &cstruct,         // base address of a structure
                           &cstruct.field1,  // position of the field in that structure
                           sizeof( cstruct.field1 ), // size of the field; 1,2,4 or 8 bytes
                           true ); //signed or unsigned data.
      \endcode

      \param data base data address for the reflective field
      \param dest address of a field in the structure
      \param size size of reflective binary data
      \param is true is signed, false if signed.
   */
   void setReflective( void *data, void *dest, uint16 size, bool is = false )
   {
      byte *p1 = (byte *) data;
      byte *p2 = (byte *) dest;
      setReflective( (uint16) (p2 - p1), size, is );
   }


   bool asBool() const { return m_value.val_bool; }
   int64 asInteger() const { return m_value.val_int; }
   const String *asString() const { return m_value.val_str; }
   const Symbol *asSymbol() const { return m_value.val_sym; }
   numeric asNumeric() const { return m_value.val_num; }
   int16 asReflectiveOffset() { return m_value.val_reflect.offset; }
   int16 asReflectiveSize() { return m_value.val_reflect.size; }
   bool asReflectiveIsSigned() { return m_value.val_reflect.isSigned; }

   bool isNil() const { return m_val_type == t_nil; }
   bool isBool() const { return m_val_type == t_bool; }
   bool isInteger() const { return m_val_type == t_int; }
   bool isString() const { return m_val_type == t_string; }
   bool isNumeric() const { return m_val_type == t_num; }
   bool isSymbol() const { return m_val_type == t_symbol || m_val_type == t_base; }
   bool isBaseClass() const { return m_val_type == t_base; }
   bool isReference() const { return m_val_type == t_reference; }
   bool isReflective() const { return m_val_type == t_reflective; }

   bool save( Stream *out ) const;
   bool load( Module *mod, Stream *in );
};


/** Implements an external function definition.
*/

class FALCON_DYN_CLASS ExtFuncDef: public BaseAlloc
{
   /** Function. */
   ext_func_t m_func;

public:
   ExtFuncDef( ext_func_t func ):
      m_func( func )
   {}

   /** Call this function.
      Will crash if function is not an external function.
   */
   void call( VMachine *vm ) const { m_func( vm ); }
   void operator()( VMachine *vm ) const { call( vm ); }
};

/** Implements a callable symbol.
   A callable symbol has many more fields to keep track of. It has a back-pointer to the owning module,
   because the module contains the bytecode where data is held. It has also a symbol table pointer
   holding variable only symbols in it.
*/
class FALCON_DYN_CLASS FuncDef: public BaseAlloc
{
   /** Private sub-symbol table. Used in classes & callables */
   SymbolTable m_symtab;

   /** Position in the file where the symbol can be called. */
   uint32 m_offset;

   /** Minimal default parameters */
   uint16 m_params;

   /** Count of local variables */
   uint16 m_locals;

   /** Count of local variables - still undefined */
   uint16 m_undefined;

   /** Cache item for functions with static elements.
      If the function is not static, it will be NO_STATE.
   */
   uint32 m_onceItemId;

public:
   enum {
		NO_STATE = 0xFFFFFFFF
	} enum_NO_STATE;

   FuncDef( uint32 offset );
   ~FuncDef();

   const SymbolTable &symtab() const { return m_symtab; }
   SymbolTable &symtab() { return m_symtab; }

   Symbol *addParameter( Symbol *sym );
   Symbol *addLocal( Symbol *sym );
   Symbol *addUndefined( Symbol *sym );

   uint32 offset() const { return m_offset; }
   void offset( uint32 o ) { m_offset = o; }
   uint16 params() const { return m_params; }
   uint16 locals() const { return m_locals; }
   uint16 undefined() const { return m_undefined; }
   void params( uint16 p ) { m_params = p; }
   void locals( uint16 l ) { m_locals = l; }
   void undefined( uint16 u ) { m_undefined = u; }

   /** Counts the parameters and the local variables that this funcdef has in its symtab.
      Useful when the function object is created by the compiler, that won't use addParameter()
      and addLocal(). Those two metods are for API implementors and extensions, while the
      compiler has to add symbols to the function symbol table; for this reason, at the end
      of the function the compiler calls this method to cache the count of locals and parameters
      that the linker must create.
   */
   void recount();

   bool save( Stream *out ) const;
   bool load( Module *mod, Stream *in );

   void onceItemId( uint32 id ) { m_onceItemId = id; }
   uint32 onceItemId() const { return m_onceItemId; }
};


/** Inheritance definition.
   Every class may be derived from one or more subclasses. Every inheritance
   will force the class to include all the properties of the subclasses, and
   to eventually call the base class constructors, if they have. Constructors
   may be called with parameters, which may be constants or symbols taken from
   the class symbol table, that is, from the parameters that have been passed
   to the class.

   This structure is needed to record the order and kind of parameters given
   to subclasses instantation without writing VM code. In this way, each class
   constructor will handle only its local instantation, and the correct call
   of each constructor will be handled outside the main VM loop, in a smart
   and efficient way.
*/
class FALCON_DYN_CLASS InheritDef: public BaseAlloc
{
private:
   /** The base class. */
   Symbol *m_baseClass;

   /** Parameter list.
      The VarDefs in this list are owned by this class.
      The destruction of this class causes destruction of the defs.
      However, Falcon::VarDef doesn't own its contents.
   */
   List m_params;

public:
   /** Constructs the Inheritance.
      The inheritance is initially built without parameters.
      \param bc the base class this inheritance refers to.
   */
   InheritDef( Symbol *bc ):
      m_baseClass( bc )
   {}

   /** Empty constructor.
      Mainly used during deserialization.
   */
   InheritDef():
      m_baseClass( 0 )
   {}

   ~InheritDef();

   void addParameter( VarDef *def );

   bool save( Stream *out ) const;
   bool load( Module *mod, Stream *in );

   const List &parameters() const { return m_params; }
   List &parameters() { return m_params; }
   Symbol *base() const { return m_baseClass; }
};

/** Class symbol abstraction.
   A class symbol has multiple lives: it looks like a function, and if used
   as a function it has the effect to create an object instance. The class
   constructor is in fact the "function" held by the symbol. Other than
   that, the class symbol stores also symbol tables for variables (properties)
   and functions (methods), other than the local + params symbol table as
   any other callable. Finally, the ClassSymbol holds a vector of direct
   parent pointers, each of which being a ClassSymbol.
*/
class FALCON_DYN_CLASS ClassDef: public FuncDef
{
public:
   typedef List InheritList;

private:

   /** Class onstructor.
      This symbol may contain a Falcon function in any module; the
      function must just be aware that it will "self" valorized to the newborn
      object, and it will be called with the same parameters by which the class
      is called.
   */
   Symbol *m_constructor;

   /** Properties table.
      The properties symbol table has not the ownership of its symbol. All symbols are
      owned by the module. If the class is somehow destroyed, its member may be still
      available to the module (i.e. they may be have already been moved to another class,
      or as standalone functions).

      This is valid also for the data members, which are named after the "class.propery"
      scheme.

      However, the strings used as the keys in this table are NOT the symbol name ("class.property"),
      but the property name that the class uses to identify a member (i.e. just "property").

      The inheritance is part of the properties, as parent classes are accesible via the
      operator "class.base_class".

      (const String *, VarDef *)
   */
   Map m_properties;

   /** Inheritance list.
   */
   InheritList m_inheritance;

   /** Startup has */
   SymbolList m_has;

   /** Startup hasnt */
   SymbolList m_hasnt;

public:
   ClassDef( uint32 offset=0, Symbol *ext_ctor=0 );
   ~ClassDef();

   const Map &properties() const { return m_properties; }
   /** Return a updateable version of the property map. */
   Map &properties() { return m_properties; }

   const InheritList &inheritance() const { return m_inheritance; }
   /** Return a updateable version of the inheritance list. */
   InheritList &inheritance() { return m_inheritance; }

   void constructor( Symbol *ctor ) { m_constructor = ctor; }
   Symbol *constructor() const { return m_constructor; }

   /** Accessor to the list of Symbol declared as "HAS" clause.
      Const version.
      \return the list of symbols that this class should have as attributes
   */
   const SymbolList &has() const { return m_has; }

   /** Accessor to the list of Symbol declared as "HAS" clause.
      \return the list of symbols that this class should have as attributes
   */
   SymbolList &has() { return m_has; }

   /** Accessor to the list of Symbol undefined in "HAS" clause.
      Const version.
      \return the list of symbols that this class won't have as attributes
   */
   const SymbolList &hasnt() const { return m_hasnt; }

   /** Accessor to the list of Symbol undefined in "HAS" clause.
      \return the list of symbols that this class won't have as attributes
   */
   SymbolList &hasnt() { return m_hasnt; }

   bool save( Stream *out ) const;
   bool load( Module *mod, Stream *in );

   /** Adds a property to a class.
      If the a property with the same name already existed,
      the content is destroyed and then overwritten.

      The name is a string pointer that should be contained
      int the module string table.

      \param name the name of the property
      \param definition the definition of the property as a VarDef
   */

   void addProperty( const String *name, VarDef *definition );

   bool hasProperty( const String &name )
   {
      return m_properties.find( &name ) != 0;
   }

   /** Shortcut to add a variable property.
      \param name the name of the property that is considered a NIL variable.
   */
   void addProperty( const String *name )
   {
      addProperty( name, new VarDef() );
   }

   /** Shortcut to add a method to a class.
      \param name the name of the method.
      \param method the Symbol holding the method call.
   */
   void addProperty( const String *name, Symbol *method )
   {
      addProperty( name, new VarDef( method ) );
   }

   VarDef *getProperty( const String *name ) const;

   VarDef *getProperty( const String &name ) const
   {
      return getProperty( &name );
   }

   /** Adds a symbol as an inheritance instance.
      It also takes care to store it in the property table.
      \return true if the add is successful, false if the inheritance has been already added.
   */
   bool addInheritance( InheritDef *parent_class );

   /** Returns true if one of the base classes of this one has the given name. */
   bool inheritsFrom( const String &find_name ) const;
};

/** Representation of a VM symbol
   For now, it is only an accessible strucutre.
   As symbol names are allocated in the symbol table part of the module,
   they are currently not deleted at item destruction. Symbol names \b must
   be allocated in the module symbol table.
*/

class FALCON_DYN_CLASS Symbol: public BaseAlloc
{
public:
   typedef enum {
      tundef,
      tglobal,
      tlocal,
      tparam,
      tlocalundef,
      tfunc,
      textfunc,
      tclass,
      tprop,
      tvar,
      tinst,
      tconst,
      tattribute
   } type_t;

private:
   typedef enum {
      FLAG_EXPORTED=0x1,
      FLAG_ETAFUNC=0x2,
      FLAG_WELLKNOWN=0x4,
      FLAG_IMPORTED=0x8,
      FLAG_ENUM=0x16
   }
   e_flags;

   type_t m_type;

   /** Flags as exported or ETA func */
   uint8 m_flags;

   /** Position of the item in the variable table. */
   uint16 m_itemPos;

   /** ID of a symbol in its module table. */
   uint32 m_id;

   /** Line at which a symbol is declared */
   int32 m_lineDecl;

   /** Symbol name; actually existing in its module definition, so not allocated nor deallocated. */
   const String *m_name;

   /** Module where the symbol resides. */
   Module *m_module;

   union {
      FuncDef *v_func;
      ExtFuncDef *v_extfunc;
      ClassDef *v_class;
      VarDef *v_prop;
      Symbol *v_symbol;
   } m_value;

   void clear();

public:

   /** Convenience constructor.
      This creates the symbol with minimal setup
      \param mod the owner module
      \param id the id of the symbol in the module
      \param name the name of the symbol
      \param exp true if this symbol is exported.
   */
   Symbol( Module *mod, uint32 id, const String *name, bool exp ):
      m_name( name ),
      m_id( id ),
      m_type( tundef ),
      m_flags(exp ? 1: 0),
      m_module(mod),
      m_lineDecl(0)
   {}

   /** Builds a symbol without ID and export class.
      The symbol is by default not exported, and it's id is left to a
      non-meaningful value. This constructor is meant to be used only
      on symbols that are going to be added somehow to a module soon after.

      The type of the symbol is Symbol::tundef.
      \param mod the owner module
      \param name a String that has been created and stored somewhere safe
                 (i.e. module string table).
   */
   Symbol( Module *mod, const String *name ):
      m_name( name ),
      m_type( tundef ),
      m_flags( 0 ),
      m_module( mod ),
      m_lineDecl(0),
      m_id(0)
   {}

   /** Basic empty constructor.
      This is mainly used for de-serialization to prepare the symbols
      before it can be de-serialized.
      \param owner the module that is owning this symbol
      \note by default, the symbol is not exported.
   */
   Symbol( Module *owner ):
      m_module( owner ),
      m_name( 0 ),
      m_id( 0 ),
      m_type( tundef ),
      m_flags( 0 ),
      m_lineDecl(0)
   {}

   ~Symbol() {
      clear();
   }

   /** Changes the symbol name.
      Symbol names are never destroyed, so the old name is not
      de-allocated. If necessary, the application must keep track of that.
      \param name The new name.
   */
   void name( const String *name ) { m_name = name; }

   /** Changes the symbol id.
      \param i the new id.
   */
   void id( uint32 i ) { m_id = i; }

   /** Sets the symbol export class.
      \param exp true if the symbol must be exported, false otherwise.
      \return itself
   */
   Symbol &exported( bool exp ) {
      if ( exp )
         m_flags |= FLAG_EXPORTED;
      else
         m_flags &=~FLAG_EXPORTED;
      return *this;
   }

   /** Sets the symbol import class.
      Import class is prioritary to export class; that is, if a symbol is imported,
      exported() will always return false. Also, imported symbols report their type as unknonw,
      no matter what local setting is provided.
      \param exp true if the symbol must be imported, false otherwise.
      \return itself
   */
   Symbol &imported( bool exp ) {
      if ( exp )
         m_flags |= FLAG_IMPORTED;
      else
         m_flags &=~FLAG_IMPORTED;
      return *this;
   }

   /** Declares the symbol as an "eta function".
      Eta functions are self-managed functions in Sigma-evaluation
      (functional evaluation).
      \param exp true if the symbol is an ETA function, false otherwise.
      \return itself
   */
   Symbol &setEta( bool exp ) {
      if ( exp )
         m_flags |= FLAG_ETAFUNC;
      else
         m_flags &=~FLAG_ETAFUNC;
      return *this;
   }

   /** Declares the symbol as an "well known symbol".

      Normal symbols are generated in a module, and eventually exported to
      the global namespace of the VM. Well known symbols live in a special
      space in the VM; they are always referenced and the module declaring
      them receives a copy of the original item, but not the original one.

      Modules can i.e. change objects and can alter functions, but a copy
      of the original well knonw items is kept by the VM and is available
      to C++ extensions.

      Well known items are meant to provide language-oriented special features,
      or to provide special hooks for modules. Error, TimeStamp and other language
      relevant classes are WKS (well known symbol). Modules can declare new
      well known symbol to i.e. declare new classes and provide C++ factory functions.

      \param exp true if the symbol is a Well Known Symbol.
      \return itself
   */
   Symbol &setWKS( bool exp ) {
      if ( exp )
         m_flags |= FLAG_WELLKNOWN;
      else
         m_flags &=~FLAG_WELLKNOWN;
      return *this;
   }


   /** Declares the symbol as an "enum class".
      Enum classes are classes that only store constant values, or serve
      otherwise as namespaces.
      They cannot be instantiated.
      \param exp true if the symbol is an enumeration.
      \return itself
   */
   Symbol &setEnum( bool exp ) {
      if ( exp )
         m_flags |= FLAG_ENUM;
      else
         m_flags &=~FLAG_ENUM;
      return *this;
   }

   void setUndefined() { clear(); m_type = tundef; }
   void setLocalUndef() { clear(); m_type = tlocalundef; }
   void setGlobal() { clear(); m_type = tglobal; }
   void setLocal() { clear(); m_type = tlocal; }
   void setParam() { clear(); m_type = tparam; }
   void setFunction( FuncDef *func ) { clear(); m_type = tfunc; m_value.v_func = func; }
   void setExtFunc( ExtFuncDef *f ) { clear(); m_type = textfunc; m_value.v_extfunc = f; }
   void setClass( ClassDef *f ) { clear(); m_type = tclass; m_value.v_class = f; }
   void setProp( VarDef *p ) { clear(); m_type = tprop; m_value.v_prop = p; }
   void setVar( VarDef *p ) { clear(); m_type = tvar; m_value.v_prop = p; }
   void setConst( VarDef *p ) { clear(); m_type = tconst; m_value.v_prop = p; }
   void setAttribute() { clear(); m_type = tattribute; }
   void setInstance( Symbol *base_class ) { clear(); m_type = tinst; m_value.v_symbol = base_class; }

   const String &name() const { return *m_name; }
   uint32 id() const { return m_id; }
   type_t type() const { return m_type; }
   bool exported() const { return (! imported()) && ((m_flags & FLAG_EXPORTED) == FLAG_EXPORTED); }
   bool imported() const { return (m_flags & FLAG_IMPORTED) == FLAG_IMPORTED; }
   uint16 itemId() const { return m_itemPos; }
   void itemId( uint16 ip ) { m_itemPos = ip; }
   bool isEta() const { return (m_flags & FLAG_ETAFUNC) == FLAG_ETAFUNC; }
   bool isWKS() const { return (m_flags & FLAG_WELLKNOWN) == FLAG_WELLKNOWN; }
   bool isEnum() const { return (m_flags & FLAG_ENUM) == FLAG_ENUM; }

   bool isUndefined() const { return imported() || m_type == tundef; }
   bool isLocalUndef() const { return m_type == tlocalundef; }
   bool isGlobal() const { return m_type == tglobal; }
   bool isLocal() const { return m_type == tlocal; }
   bool isParam() const { return m_type == tparam; }
   bool isFunction() const { return m_type == tfunc; }
   bool isExtFunc() const { return m_type == textfunc; }
   bool isClass() const { return m_type == tclass; }
   bool isProp() const { return m_type == tprop; }
   bool isVar() const { return m_type == tvar; }
   bool isCallable() const { return isFunction() || isExtFunc() || isClass(); }
   bool isInstance() const { return m_type == tinst; }
   bool isConst() const { return m_type == tconst; }
   bool isAttribute() const  { return m_type == tconst; }

   FuncDef *getFuncDef() const { return m_value.v_func; }
   ExtFuncDef *getExtFuncDef() const { return m_value.v_extfunc; }
   ClassDef *getClassDef() const { return m_value.v_class; }
   VarDef *getVarDef() const { return m_value.v_prop; }
   Symbol *getInstance() const { return m_value.v_symbol; }
   uint16 getItemId() const { return m_itemPos; }

   /** If the symbol is a class, check if this class is the named one or derived from the named one.
      If the name passed as parameters is the name of this class or
      one of its ancestor names, the function returns true.
      \param find_name The class name.
   */
   bool fromClass( const String &find_name ) const;

   bool save( Stream *out ) const;
   bool load( Stream *in );

   /** Returns the module associated with a symbol.
      \return the owner module or zero if the information is not available.
   */
   const Module *module() const { return m_module; }
   void module( Module *mod ) { m_module = mod; }

   void declaredAt( int32 line ) { m_lineDecl = line; }
   int32 declaredAt() const { return m_lineDecl; }
};


/** maybe not elegant, but a safe way to store the kind of pointer. */
#define FLC_CLSYM_VAR ((Falcon::Symbol *) 0)
#define FLC_CLSYM_METHOD ((Falcon::Symbol *) 1)
#define FLC_CLSYM_BASE ((Falcon::Symbol *) 2)

}

#endif
/* end of flc_symbol.h */

