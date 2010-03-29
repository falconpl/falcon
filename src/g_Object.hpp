#ifndef G_OBJECT_HPP
#define G_OBJECT_HPP

#include "modgtk.hpp"


namespace Falcon {
namespace Glib {

/**
 *  \class Falcon::Glib::Object
 */
class Object
    :
    public Gtk::CoreGObject
{
public:

    Object( const Falcon::CoreClass*, const GObject* = 0 );

    static Falcon::CoreObject* factory( const Falcon::CoreClass*, void*, bool );

    static void modInit( Falcon::Module* );

    static FALCON_FUNC signal_notify( VMARG );

    static void on_notify( GObject*, GParamSpec*, gpointer );

    static FALCON_FUNC set_property( VMARG );

    static FALCON_FUNC get_property( VMARG );

    static FALCON_FUNC notify( VMARG );

    static FALCON_FUNC freeze_notify( VMARG );

    static FALCON_FUNC thaw_notify( VMARG );

    static FALCON_FUNC ref( VMARG );

    static FALCON_FUNC unref( VMARG );

    static FALCON_FUNC ref_sink( VMARG );

};


} // Glib
} // Falcon

#endif // !G_OBJECT_HPP
