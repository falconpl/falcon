#ifndef MODGTK_HPP
#define MODGTK_HPP

#include <falcon/autocstring.h>
#include <falcon/coreobject.h>
#include <falcon/coreslot.h>
#include <falcon/error.h>
#include <falcon/falcondata.h>
#include <falcon/garbagelock.h>
#include <falcon/item.h>
#include <falcon/module.h>
#include <falcon/stream.h>
#include <falcon/string.h>
#include <falcon/vm.h>

#include <glib-object.h>

#include <assert.h>
#include <stdio.h>
#include <string.h>

#include "modgtk_st.hpp"
#include "modgtk_version.hpp"

/*
 *  some helper defines..
 */

#define VMARG           ::Falcon::VMachine* vm

#define MYSELF          ::Falcon::CoreObject* self = vm->self().asObject()

#define GET_OBJ( self ) \
        GObject* _obj = ((::Falcon::Gtk::GData*) self->getUserData())->obj()

#define GET_SIGNALS( gobj ) \
        ::Falcon::CoreSlot* _signals = (::Falcon::CoreSlot*) \
        g_object_get_data( G_OBJECT( gobj ), "_signals" )

#define IS_DERIVED( it, cls ) \
        ( (it)->isOfClass( #cls ) || (it)->isOfClass( "gtk." #cls ) )

#define CoreObject_IS_DERIVED( obj, cls ) \
        ( (obj)->derivedFrom( #cls ) || (obj)->derivedFrom( "gtk." #cls ) )

#define throw_inv_params( x ) \
        throw new ::Falcon::ParamError( \
        ::Falcon::ErrorParam( ::Falcon::e_inv_params, __LINE__ ).extra( x ) )

#define throw_require_no_args() \
        throw_inv_params( FAL_STR( gtk_e_require_no_args_ ) )

#define throw_gtk_error( n, x ) \
        throw new ::Falcon::Gtk::GtkError( \
        ::Falcon::ErrorParam( ::Falcon::Gtk::n, __LINE__ ).desc( x ) )


namespace Falcon {

/**
 *  \namespace Falcon::Gtk
 */
namespace Gtk {


/**
 *  \class Falcon::Gtk::GData
 */
class GData
    :
    public Falcon::FalconData
{

    GObject*    m_obj;

public:

    GData( GObject* obj )
        :
        m_obj( obj )
    {
        incref();
    }

    ~GData() { decref(); }

    void gcMark( Falcon::uint32 ) {}

    FalconData* clone() const { return 0; }

    void incref() { if ( m_obj ) g_object_ref_sink( m_obj ); }

    GData* increfed() { incref(); return this; }

    void decref() { if ( m_obj ) g_object_unref( m_obj ); }

    GObject* obj() const { return m_obj; }

};


/**
 *  \brief install a slot inside user data
 */
GObject* internal_add_slot( GObject* );


/**
 *  \brief release internal slots
 *  Function of type GDestroyNotify to delete internal slots
 */
void internal_release_slot( gpointer );


/**
 *  \brief get internal slot
 *  \param signame signal name
 *  \param cb callback function
 *  \param vm virtual machine
 */
void internal_get_slot( const char* signame, void* cb, Falcon::VMachine* vm );


/**
 *  \brief trigger internal slot
 *  \param obj signal emitter
 *  \param signame signal name
 *  \param cbname callback name
 *  \param vm virtual machine
 */
void internal_trigger_slot( GObject* obj, const char* signame,
        const char* cbname, Falcon::VMachine* vm );


/**
 *  \brief common init method for all abstract classes
 */
FALCON_FUNC abstract_init( VMARG );


/**
 *  \brief get a signal
 *  \param obj signal emitter
 *  \param sig emitter's signals container
 *  \param name signal name
 *  \param cb callback function
 *  \param vm current virtual machine
 *  \return the slot activated by the named signal
 */
Falcon::CoreSlot* get_signal( GObject* obj, Falcon::CoreSlot* sig,
                              const char* name, void* cb, Falcon::VMachine* vm );


/**
 *  \class Falcon::Gtk::CoreGObject
 *  \brief base class for g-items derived from Falcon::CoreObject
 */
class CoreGObject
    :
    public Falcon::CoreObject
{
public:

    CoreGObject( const Falcon::CoreClass* cls )
        :
        Falcon::CoreObject( cls )
    {}

    ~CoreGObject() {}

    Falcon::CoreObject* clone() const { return 0; }

    bool getProperty( const Falcon::String&, Falcon::Item& ) const;

    bool setProperty( const Falcon::String&, const Falcon::Item& );

    static void delProperty( gpointer );

};


/**
 *  \class Falcon::Gtk::GtkError
 */
class GtkError
    :
    public Falcon::Error
{
public:

    GtkError()
        :
        Error( "GtkError" )
    {}

    GtkError( const ErrorParam& params  )
        :
        Error( "GtkError", params )
    {}

};


/**
 *  \brief exception type initialization
 */
FALCON_FUNC GtkError_init ( VMARG );


/**
 *  \enum Falcon::Gtk::GtkErrorIds
 */
enum GtkErrorIds
{
    e_abstract_class,       // unable to create instance of abstract type
    e_init_failure,         // failure due to gtk_init* functions
    e_inv_property          // invalid property
};


/**
 *  \brief struct holding class methods information
 */
typedef struct
{
    const char* name;
    void (*cb)( Falcon::VMachine* );
} MethodTab;


/**
 *  \brief struct holding enums information
 */
typedef struct
{
    const char* name;
    Falcon::int64 value;
} ConstIntTab;


/**
 *  \brief class to help with arguments checking
 */
template<int numStrings>
class ArgCheck
{

    Falcon::AutoCString m_strings[ numStrings ];

    Falcon::VMachine*   m_vm;

    const char*     m_spec;

    int     m_p;

public:

    ArgCheck( Falcon::VMachine* vm, const char* spec )
        :
        m_vm( vm ),
        m_spec( spec ),
        m_p( 0 )
    {}

    char* getCString( int index, bool mandatory = true )
    {
        Item* it = m_vm->param( index );
        if ( mandatory )
        {
#ifndef NO_PARAMETER_CHECK
            if ( !it || it->isNil() || !it->isString() )
                throw_inv_params( m_spec );
#endif
        }
        else
        {
            if ( !it || it->isNil() )
                return 0;
#ifndef NO_PARAMETER_CHECK
            if ( !it->isString() )
                throw_inv_params( m_spec );
#endif
        }
        m_strings[ m_p ].set( it->asString() );
        return (char*) m_strings[ m_p++ ].c_str();
    }

    Falcon::int64 getInteger( int index, bool mandatory = true, bool* wasNil = 0 ) const
    {
        Item* it = m_vm->param( index );
        if ( mandatory )
        {
#ifndef NO_PARAMETER_CHECK
            if ( !it || it->isNil() || !it->isInteger() )
                throw_inv_params( m_spec );
#endif
        }
        else
        {
            if ( !it || it->isNil() )
            {
                if ( wasNil )
                    *wasNil = true;
                return 0;
            }
#ifndef NO_PARAMETER_CHECK
            if ( !it->isInteger() )
                throw_inv_params( m_spec );
#endif
            if ( wasNil )
                *wasNil = false;
        }
        return it->asInteger();
    }

    Falcon::numeric getNumeric( int index, bool mandatory = true, bool* wasNil = 0 ) const
    {
        Item* it = m_vm->param( index );
        if ( mandatory )
        {
#ifndef NO_PARAMETER_CHECK
            if ( !it || it->isNil() || !it->isOrdinal() )
                throw_inv_params( m_spec );
#endif
        }
        else
        {
            if ( !it || it->isNil() )
            {
                if ( wasNil )
                    *wasNil = true;
                return 0;
            }
#ifndef NO_PARAMETER_CHECK
            if ( !it->isOrdinal() )
                throw_inv_params( m_spec );
#endif
            if ( wasNil )
                *wasNil = false;
        }
        return it->asNumeric();
    }

    bool getBoolean( int index, bool mandatory = true, bool* wasNil = 0 ) const
    {
        Item* it = m_vm->param( index );
        if ( mandatory )
        {
#ifndef NO_PARAMETER_CHECK
            if ( !it || it->isNil() || !it->isBoolean() )
                throw_inv_params( m_spec );
#endif
        }
        else
        {
            if ( !it || it->isNil() )
            {
                if ( wasNil )
                    *wasNil = true;
                return false;
            }
#ifndef NO_PARAMETER_CHECK
            if ( !it->isBoolean() )
                throw_inv_params( m_spec );
#endif
            if ( wasNil )
                *wasNil = false;
        }
        return it->asBoolean();
    }

    Falcon::CoreObject* getObject( int index, bool mandatory = true ) const
    {
        Item* it = m_vm->param( index );
        if ( mandatory )
        {
#ifndef NO_PARAMETER_CHECK
            if ( !it || it->isNil() || !it->isObject() )
                throw_inv_params( m_spec );
#endif
        }
        else
        {
            if ( !it || it->isNil() )
                return 0;
#ifndef NO_PARAMETER_CHECK
            if ( !it->isObject() )
                throw_inv_params( m_spec );
#endif
        }
        return it->asObject();
    }

    ~ArgCheck() {}

};


/*
 *  typedefs for Gtk::ArgCheck<x>
 */

#ifndef __GNUC__
typedef Falcon::Gtk::ArgCheck<1>    ArgCheck0;
#else
typedef Falcon::Gtk::ArgCheck<0>    ArgCheck0;
#endif
typedef Falcon::Gtk::ArgCheck<1>    ArgCheck1;
typedef Falcon::Gtk::ArgCheck<2>    ArgCheck2;
typedef Falcon::Gtk::ArgCheck<3>    ArgCheck3;
typedef Falcon::Gtk::ArgCheck<4>    ArgCheck4;
typedef Falcon::Gtk::ArgCheck<5>    ArgCheck5;
typedef Falcon::Gtk::ArgCheck<6>    ArgCheck6;
typedef Falcon::Gtk::ArgCheck<7>    ArgCheck7;
typedef Falcon::Gtk::ArgCheck<8>    ArgCheck8;
typedef Falcon::Gtk::ArgCheck<9>    ArgCheck9;

} // Gtk
} // Falcon

#endif // !MODGTK_HPP
