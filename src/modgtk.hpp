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
    e_init_failure          // failure due to gtk_init* functions

};



} // Gtk
} // Falcon

#endif // !MODGTK_HPP
