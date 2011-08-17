/*
   FALCON - The Falcon Programming Language.
   FILE: symbol.h

   Symbol -- load time static definitions for program-wide entities.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: mar ago 3 2004

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef FLC_SYMBOL_H
#define FLC_SYMBOL_H

#include <falcon/types.h>
#include <falcon/symtab.h>
#include <falcon/symlist.h>
#include <falcon/genericmap.h>
#include <falcon/basealloc.h>
#include <falcon/reflectfunc.h>
#include <falcon/objectfactory.h>
#include <falcon/fassert.h>
#include <falcon/itemid.h>
#include <falcon/vardef.h>

namespace Falcon {

class Symbol;
class Stream;
class AttribMap;

/** Implements an external function definition.
*/

class FALCON_DYN_CLASS ExtFuncDef: public BaseAlloc
{
   /** Function. */
   ext_func_t m_func;

   /** Extra data.
      \see extra()
   */
   void *m_extra;

   SymbolTable *m_params;

public:
   ExtFuncDef( ext_func_t func ):
      m_func( func ),
      m_extra( 0 ),
      m_params( 0 )
   {}

   /** Crates this definition setting extra data.
      \see extra()
   */
   ExtFuncDef( ext_func_t func, void *extra ):
      m_func( func ),
      m_extra( extra ),
      m_params(0)
   {}

   ~ExtFuncDef();

   /** Call this function.
      Will crash if function is not an external function.
   */
   void call( VMachine *vm ) const { m_func( vm ); }

    /** Call this function.
   */
   void operator()( VMachine *vm ) const { call( vm ); }

   /** Gets extra data.
      \see void extra( void *)
   */
   void *extra() const { return m_extra; }

   /** Sets extra data for this function call.

      This extra data is useful to create flexible reflected calls.
      The ext_func_t gets called with the VM data; it can then
      decode the VM data and prepare it accordingly to m_extra,
      and finally call some different function as retreived
      data instructs to do. This allows to reuse a single
      ext_func_t in m_func to call different binding function.

      This item doesn't own m_extra; the data must be held
      in the same module in which this function exists (or in the
      application where this module is run), and
      must stay valid for all the time this function stays valid.

      Extra data will be available to m_func through VMachine::symbol(),
      which will return the symbol containing this funcdef.
   */
   void extra( void *e )  { m_extra = e; }

   /** Returns the ID of the given function parameter, or -1 if the parameter doesn't exist. */
   int32 getParam( const String &name );

   /** Adds a function parameter with the specified ID.
      Consider using Symbol::addParam() instead (candy grammar).
   */
   ExtFuncDef &addParam( Symbol *param, int32 id =-1 );

   /** External Function symbol table.

      Not all the external functions need to be provided with a
      symbol table (actually storing only formal parameters).
      For this reason, the symbol table of external functions
      gets allocated
      only when actually adding parameters.
   */
   SymbolTable *parameters() const { return m_params; }

   uint32 paramCount() const { return m_params == 0 ? 0 : m_params->size(); }

   ext_func_t func() const { return m_func; }
};


/** Implements an imported symbol.
*/

class FALCON_DYN_CLASS ImportAlias: public BaseAlloc
{
   const String m_name;
   const String m_origModule;
   bool m_isFileName;

public:
   ImportAlias( const String& name, const String& origModule, bool bIsFileName = false ):
      m_name( name ),
      m_origModule( origModule ),
      m_isFileName( bIsFileName )
   {}

   const String &name() const { return m_name; }
   const String &origModule() const { return m_origModule; }
   bool isOrigFileName() const { return m_isFileName; }
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

   /** Function code.
      Owned by the symbol and destroyed on exit.
   */
   byte *m_code;

   /** Function size of the code.
      Owned by the symbol and destroyed on exit.
   */
   uint32 m_codeSize;

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

   uint32 m_basePC;

    /** Attributes table.
      It's a map of attributes. They're symbol-specific dictionaries
      of strings-values that become read-only in realtime.
      (String& , VarDef *)
   */
   AttribMap* m_attributes;

public:
   enum {
      NO_STATE = 0xFFFFFFFF
   } enum_NO_STATE;

   /** Constructor for external generators.
      Requires that the funcdef is provided with a previously allocated code.
      The code is owned by the FuncDef and destroyed with this instance.
   */
   FuncDef( byte *code, uint32 codeSize );
   ~FuncDef();

   const SymbolTable &symtab() const { return m_symtab; }
   SymbolTable &symtab() { return m_symtab; }

   Symbol *addParameter( Symbol *sym );
   Symbol *addLocal( Symbol *sym );
   Symbol *addUndefined( Symbol *sym );

   uint32 codeSize() const { return m_codeSize; }
   void codeSize( uint32 p ) { m_codeSize = p; }
   byte *code() const { return m_code; }
   void code( byte *b ) { m_code = b; }
   uint16 params() const { return m_params; }
   uint16 locals() const { return m_locals; }
   uint16 undefined() const { return m_undefined; }
   void params( uint16 p ) { m_params = p; }
   void locals( uint16 l ) { m_locals = l; }
   void undefined( uint16 u ) { m_undefined = u; }

   void basePC( uint32 pc ) { m_basePC = pc; }

   /** BasePC at which this symbol was declared.
      This is useful only to find lines in the code of this function
      in the global line table.

      The line that generated a certain function is found through
      basepc + VM->pc.
   */
   uint32 basePC() const { return m_basePC; }

   /** Counts the parameters and the local variables that this funcdef has in its symtab.
      Useful when the function object is created by the compiler, that won't use addParameter()
      and addLocal(). Those two metods are for API implementors and extensions, while the
      compiler has to add symbols to the function symbol table; for this reason, at the end
      of the function the compiler calls this method to cache the count of locals and parameters
      that the linker must create.
   */
   void recount();

   bool save( const Module *mod, Stream *out ) const;
   bool load( const Module *mod, Stream *in );

   void onceItemId( uint32 id ) { m_onceItemId = id; }
   uint32 onceItemId() const { return m_onceItemId; }

   /** Adds an attribute.
      The first attribute added will cause the map to be created.
   */
   void addAttrib( const String& name, VarDef* value );

   /** Returns a map of String& -> VarDef* containing metadata about this symbol.
      May return if the item has no attributes.
   */
   AttribMap* attributes() const  { return m_attributes; }
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

   bool save( Stream *out ) const;
   bool load( const Module *mod, Stream *in );

   Symbol *base() const { return m_baseClass; }
};

class StateDef: public BaseAlloc
{
   const String* m_name;

   /** Functions in this state.
    * (String*, Symbol*)
    * Strings and symbol are owned by the module.
   */
   Map m_functions;

public:
   StateDef( const String* sname );

   const String& name() const { return *m_name; }
   const Map& functions() const { return m_functions; }
   Map& functions() { return m_functions; }
   bool addFunction( const String& name, Symbol* func );

   bool load( const Module *mod, Stream* in );
   bool save( Stream* out ) const;
};

/** Class symbol abstraction.

   A class symbol has multiple lives: it looks like a function, and if used
   as a function it has the effect to create an object instance. The class
   constructor is in fact the "function" held by the symbol. Other than
   that, the class symbol stores also symbol tables for variables (properties)
   and functions (methods), other than the local + params symbol table as
   any other callable. Finally, the ClassSymbol holds a vector of direct
   parent pointers, each of which being a ClassSymbol.

   Class symbols can be "reflective". They can be linked with a "user data manager".

   As reflection status cannot be properly serialized (atm), only user-provided and binary
   modules can declare reflective classes.
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

      This is valid also for the data members, which are named after the "class.property"
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

   /** Object factory used by this class.
      Function used to create instances of objects from this class.
   */
   ObjectFactory m_factory;

   int m_metaclassFor;

   bool m_bFinal;

   /** State table.
         (const String *, StateDef *)
   */
   Map m_states;

public:
   /** Creates a class definition without a constructor.
      This constructor creates a class definition for those classes which doesn't
      require an "init" method declared at Falcon level, nor an external C method
      used to provide an "init" feature.

      \param manager The object factory used to manipulate externally provided user_data of this class.
         If not provided, the factory will depend on added properties.
   */
   ClassDef( ObjectFactory factory=0 );

   /** Creates the definition of this class an external constructor.

      This constructor creates a class that has an init method provided by an
      external symbol (a C function).

      \param offset the start offset of the constructor, if this class as a Falcon code constructor.
      \param manager The object factory used to manipulate externally provided user_data of this class.
         If not provided, the factory will depend on added properties.
   */
   ClassDef( Symbol *ext_ctor, ObjectFactory factory=0 );
   ~ClassDef();

   /** Declares an object manager for this class.
      The instance of the object manager should be allocated in the user
      application or in the binary modules. The classes never dispose
      them.
   */
   void factory( ObjectFactory om ) { m_factory = om; }
   ObjectFactory factory() const { return m_factory; }

   const Map &properties() const { return m_properties; }
   /** Return a updateable version of the property map. */
   Map &properties() { return m_properties; }

   const InheritList &inheritance() const { return m_inheritance; }
   /** Return a updateable version of the inheritance list. */
   InheritList &inheritance() { return m_inheritance; }

   void constructor( Symbol *ctor ) { m_constructor = ctor; }
   Symbol *constructor() const { return m_constructor; }


   bool save( const Module* mod, Stream *out ) const;
   bool load( const Module* mod, Stream *in );

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

   /** Checks if a given children of this class compares in the ancestor list. */
   bool checkCircularInheritance( const Symbol *child ) const;

   /** Returns true if one of the base classes of this one has the given name. */
   bool inheritsFrom( const String &find_name ) const;

   int isMetaclassFor() const { return m_metaclassFor; }
   void setMetaclassFor( int ItemID ) { fassert( ItemID >= 0 && ItemID < FLC_ITEM_COUNT ); m_metaclassFor = ItemID; }

   /** Set this as a "final" class.
      Final classes can be inherited, but it is not possible to inherit
      from more than one final class in the whole hierarchy.

      Final classes store binary data that must be uniquely identified
      by functions in the hierarchy.
   */
   void setFinal( bool mod )  { m_bFinal = mod; }

   /** Returns true if this class is final.
   \see setFinal()
   */
   bool isFinal() const { return m_bFinal; }

   /** Creates a new state for this class.

      States are set of functions that are applied all
      at the same time to an object. Also, special
      __enter and __leave functions are called back
      with the name of the state from which this
      state is entered or where this state is
      going to.

      An object enters the "init" state, if provided,
      after complete instantation.
   */
   StateDef* addState( const String* stateName );
   bool addState( const String* stateName, StateDef* state );

   const Map& states() const { return m_states; }
   Map& states() { return m_states; }
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
      timportalias
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
   String m_name;

   /** Module where the symbol resides. */
   Module *m_module;

   union {
      FuncDef *v_func;
      ExtFuncDef *v_extfunc;
      ImportAlias  *v_importalias;
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
   Symbol( Module *mod, uint32 id, const String &name, bool exp ):
      m_type( tundef ),
      m_flags(exp ? 1: 0),
      m_id( id ),
      m_lineDecl(0),
      m_name( name ),
      m_module(mod)
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
   Symbol( Module *mod, const String &name ):
      m_type( tundef ),
      m_flags( 0 ),
      m_id(0),
      m_lineDecl(0),
      m_name( name ),
      m_module( mod )
   {}

   /** Basic empty constructor.
      This is mainly used for de-serialization to prepare the symbols
      before it can be de-serialized.
      \param owner the module that is owning this symbol
      \note by default, the symbol is not exported.
   */
   Symbol( Module *owner ):
      m_type( tundef ),
      m_flags( 0 ),
      m_id( 0 ),
      m_lineDecl(0),
      m_module( owner )
   {}

   ~Symbol() {
      clear();
   }

   /** Changes the symbol name.
      Symbol names are never destroyed, so the old name is not
      de-allocated. If necessary, the application must keep track of that.
      \param name The new name.
   */
   void name( const String &name ) { m_name = name; }

   /** Changes the symbol id.
      \param i the new id.
   */
   void id( uint32 i ) { m_id = i; }

   /** Sets the symbol export class.
      \param exp true if the symbol must be exported, false otherwise.
      \return itself
   */
   Symbol* exported( bool exp ) {
      if ( exp )
         m_flags |= FLAG_EXPORTED;
      else
         m_flags &=~FLAG_EXPORTED;
      return this;
   }

   /** Sets the symbol import class.
      Import class has higher priority than export class; that is, if a symbol is imported
      exported() will always return false. Also, imported symbols report their type as unknown,
      no matter what local setting is provided.
      \param exp true if the symbol must be imported, false otherwise.
      \return itself
   */
   Symbol* imported( bool exp ) {
      if ( exp )
         m_flags |= FLAG_IMPORTED;
      else
         m_flags &=~FLAG_IMPORTED;
      return this;
   }

   /** Declares the symbol as an "eta function".
      Eta functions are self-managed functions in Sigma-evaluation
      (functional evaluation).
      \param exp true if the symbol is an ETA function, false otherwise.
      \return itself
   */
   Symbol* setEta( bool exp ) {
      if ( exp )
         m_flags |= FLAG_ETAFUNC;
      else
         m_flags &=~FLAG_ETAFUNC;
      return this;
   }

   /** Declares the symbol as an "well known symbol".

      Normal symbols are generated in a module, and eventually exported to
      the global namespace of the VM. Well known symbols live in a special
      space in the VM; they are always referenced and the module declaring
      them receives a copy of the original item, but not the original one.

      Modules can i.e. change objects and can alter functions, but a copy
      of the original well known items is kept by the VM and is available
      to C++ extensions.

      Well known items are meant to provide language-oriented special features,
      or to provide special hooks for modules. Error, TimeStamp and other language
      relevant classes are WKS (well known symbol). Modules can declare new
      well known symbol to i.e. declare new classes and provide C++ factory functions.

      \param exp true if the symbol is a Well Known Symbol.
      \return itself
   */
   Symbol* setWKS( bool exp ) {
      if ( exp )
         m_flags |= FLAG_WELLKNOWN;
      else
         m_flags &=~FLAG_WELLKNOWN;
      return this;
   }


   /** Declares the symbol as an "enum class".
      Enum classes are classes that only store constant values, or serve
      otherwise as namespaces.
      They cannot be instantiated.
      \param exp true if the symbol is an enumeration.
      \return itself
   */
   Symbol* setEnum( bool exp ) {
      if ( exp )
         m_flags |= FLAG_ENUM;
      else
         m_flags &=~FLAG_ENUM;
      return this;
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
   void setInstance( Symbol *base_class ) { clear(); m_type = tinst; m_value.v_symbol = base_class; }
   void setImportAlias( ImportAlias* alias )
      { clear(); m_type = timportalias; m_value.v_importalias = alias; imported(true); }
   void setImportAlias( const String &name, const String& origModule, bool bIsFileName = false ) {
      setImportAlias( new ImportAlias( name, origModule, bIsFileName ) ); }

   const String &name() const { return m_name; }

   // TODO: Remove this
   String &name() { return m_name; }

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
   bool isImportAlias() const { return m_type == timportalias; }

   /** Candy grammar to add a parameter to a function (internal or external) */
   Symbol* addParam( const String &param );

   FuncDef *getFuncDef() const { return m_value.v_func; }
   ExtFuncDef *getExtFuncDef() const { return m_value.v_extfunc; }
   ClassDef *getClassDef() const { return m_value.v_class; }
   VarDef *getVarDef() const { return m_value.v_prop; }
   Symbol *getInstance() const { return m_value.v_symbol; }
   ImportAlias* getImportAlias() const { return m_value.v_importalias; }
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

