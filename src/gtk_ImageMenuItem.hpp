#ifndef GTK_IMAGEMENUITEM_HPP
#define GTK_IMAGEMENUITEM_HPP

#include "modgtk.hpp"


namespace Falcon {
namespace Gtk {

/**
 *  \class Falcon::Gtk::ImageMenuItem
 */
class ImageMenuItem
    :
    public Gtk::CoreGObject
{
public:

    ImageMenuItem( const Falcon::CoreClass*, const GtkImageMenuItem* = 0 );

    static Falcon::CoreObject* factory( const Falcon::CoreClass*, void*, bool );

    static void modInit( Falcon::Module* );

    static FALCON_FUNC init( VMARG );

    static FALCON_FUNC set_image( VMARG );

    static FALCON_FUNC get_image( VMARG );

    static FALCON_FUNC new_from_stock( VMARG );

    static FALCON_FUNC new_with_label( VMARG );

    static FALCON_FUNC new_with_mnemonic( VMARG );

#if GTK_MINOR_VERSION >= 16
    static FALCON_FUNC get_use_stock( VMARG );

    static FALCON_FUNC set_use_stock( VMARG );

    static FALCON_FUNC get_always_show_image( VMARG );

    static FALCON_FUNC set_always_show_image( VMARG );

    //static FALCON_FUNC set_accel_group( VMARG );
#endif

};


} // Gtk
} // Falcon

#endif // !GTK_IMAGEMENUITEM_HPP
