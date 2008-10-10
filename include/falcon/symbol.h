/*
   FALCON - The Falcon Programming Language.
   FILE: flc_symbol.h

   Short description
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
#include <falcon/objectmanager.h>
#include <falcon/fassert.h>

namespace Falcon {

class Symbol;
class Stream;
class ObjectManager;

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

   bool save( Stream *out ) const;
   bool load( Module *mod, Stream *in );
};


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
      which will return the symbol containig this funcdef.
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

   Class symbols can be "reflective". They can be linked with a "user data manager" or
   "ObjectManager" which creates, destroys and manages externally provided data for the objects.

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

   /** Object manager used by this class.
      If this class is reflective it NEEDS to set up an object manager that
      will handle the reflective data provided by the external program.

      Reflective class cannot be extended from more than another reflective
      class. It's impossible to link two different reflective classes in the
      same subclass.
   */
   ObjectManager *m_manager;

public:
   /** Creates a class definition without a constructor.
      This constructor creates a class definition for those classes which doesn't
      require an "init" method declared at Falcon level, nor an external C method
      used to provide an "init" feature.

      \param manager The object manager used to manipulate externally provided user_data of this class;
         if not provided, this class will be fully non-reflexive (and providing some vardef
         with reflexivity enabled will lead to undefined results).
   */
   ClassDef( ObjectManager *manager=0 );

   /** Creates the definition of this class an external constructor.

      This constructor creates a class that has an init method provided by an
      external symbol (a C function).

      \param offset the start offset of the constructor, if this class as a Falcon code constructor.
      \param manager The object manager used to manipulate externally provided user_data of this class;
         if not provided, this class will be fully non-reflexive (and providing some vardef
         with reflexivity enabled will lead to undefined results).
   */
   ClassDef( Symbol *ext_ctor, ObjectManager *manager=0 );
   ~ClassDef();

   /** Declares an object manager for this class.
      The instance of the object manager should be allocated in the user
      application or in the binary modules. The classes never dispose
      them.
   */
   void setObjectManager( ObjectManager *om ) { m_manager = om; }

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

   /** Returns the object manager used by this class.
      If this class has an object manager, then is a "reflective class", and can
      handle dynamically objects (create and destroy them).

      See the object manager for further details.
   */
   ObjectManager *getObjectManager() const { return m_manager; }

   /** Set this ClassDef as using a UserData reflection model.
      This method sets int this class definition the UserData ObjectManager,
      which is meant to provide set/get properties callbacks.

      This means that all the instances of this class should receive a
      UserData in the user-provided init method. It is legal to give or
      set 0 as the user data at any point in the lifetime of the class.

      \see UserData
      \see UserDataManager

      \note This is a shortcut to setting Falcon::core_user_data_cacheful as the
         object manager for this class.
   */
   void carryUserData();

   /** Set this ClassDef as using a UserData (without cache) reflection model.
      This method sets int this class definition the UserData ObjectManager,
      which is meant to provide set/get properties callbacks.

      This means that all the instances of this class should receive a
      UserData in the user-provided init method. It is legal to give or
      set 0 as the user data at any point in the lifetime of the class.

      \see UserData
      \see UserDataManager

      \note This is a shortcut to setting Falcon::core_user_data_cacheless as the
         object manager for this class.

      This version of the method prevents the final class to have a local cache.
      This means that 1) either the class don't need a location where to store
      temporarily created Falcon items or 2) the class provides a cache for some
      property internally.
   */
   void carryUserDataCacheless();

   /** Sets this class as carrying falcon data.
      FalconData reflective model is lighter than UserData model, and is
      used internally by the engine to reflect classes more precisely.
      \see FalconData
      \see FalconDataManager
   */
   void carryFalconData();
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
      const String *m_extModName;
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
   Symbol( Module *mod, const String *name ):
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
      m_name( 0 ),
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
   void name( const String *name ) { m_name = name; }

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
      Import class is prioritary to export class; that is, if a symbol is imported,
      exported() will always return false. Also, imported symbols report their type as unknonw,
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
      of the original well knonw items is kept by the VM and is available
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

   /** Candy grammar to add a parameter to a function (internal or external) */
   Symbol* addParam( const String &param );

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

   /** Shortcut to ClassDef::carryUserData. */
   void carryUserData() {
      fassert( isClass() );
      getClassDef()->carryUserData();
   }

   /** Shortcut to ClassDef::carryUserDataCacheless
   */
   void carryUserDataCacheless() {
      fassert( isClass() );
      getClassDef()->carryUserDataCacheless();
   }

   /** Shortcut to ClassDef::carryFalconData
   */
   void carryFalconData(){
      fassert( isClass() );
      getClassDef()->carryFalconData();
   }
};


/** maybe not elegant, but a safe way to store the kind of pointer. */
#define FLC_CLSYM_VAR ((Falcon::Symbol *) 0)
#define FLC_CLSYM_METHOD ((Falcon::Symbol *) 1)
#define FLC_CLSYM_BASE ((Falcon::Symbol *) 2)

}

#endif
/* end of flc_symbol.h */

