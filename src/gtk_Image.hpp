#ifndef GTK_IMAGE_HPP
#define GTK_IMAGE_HPP

#include "modgtk.hpp"


namespace Falcon {
namespace Gtk {

/**
 *  \class Falcon::Gtk::Image
 */
class Image
    :
    public Gtk::CoreGObject
{
public:

    Image( const Falcon::CoreClass*, const GtkImage* = 0 );

    static Falcon::CoreObject* factory( const Falcon::CoreClass*, void*, bool );

    static void modInit( Falcon::Module* );

    static FALCON_FUNC init( VMARG );

    //static FALCON_FUNC get_icon_set( VMARG );

    //static FALCON_FUNC get_image( VMARG );

    //static FALCON_FUNC get_pixbuf( VMARG );

    //static FALCON_FUNC get_pixmap( VMARG );

    //static FALCON_FUNC get_stock( VMARG );

    //static FALCON_FUNC get_animation( VMARG );

    //static FALCON_FUNC get_icon_name( VMARG );

    //static FALCON_FUNC get_gicon( VMARG );

    //static FALCON_FUNC get_storage_type( VMARG );

    //static FALCON_FUNC new_from_file( VMARG );

    //static FALCON_FUNC new_from_icon_set( VMARG );

    //static FALCON_FUNC new_from_image( VMARG );

    //static FALCON_FUNC new_from_pixbuf( VMARG );

    //static FALCON_FUNC new_from_pixmap( VMARG );

    static FALCON_FUNC new_from_stock( VMARG );

    //static FALCON_FUNC new_from_animation( VMARG );

    //static FALCON_FUNC new_from_icon_name( VMARG );

    //static FALCON_FUNC new_from_gicon( VMARG );

    static FALCON_FUNC set_from_file( VMARG );

    //static FALCON_FUNC set_from_icon_set( VMARG );

    //static FALCON_FUNC set_from_image( VMARG );

    //static FALCON_FUNC set_from_pixbuf( VMARG );

    //static FALCON_FUNC set_from_pixmap( VMARG );

    static FALCON_FUNC set_from_stock( VMARG );

    //static FALCON_FUNC set_from_animation( VMARG );

    //static FALCON_FUNC set_from_icon_name( VMARG );

    //static FALCON_FUNC set_from_gicon( VMARG );

    static FALCON_FUNC clear( VMARG );

    //static FALCON_FUNC set( VMARG );

    //static FALCON_FUNC get( VMARG );

    //static FALCON_FUNC set_pixel_size( VMARG );

    //static FALCON_FUNC get_pixel_size( VMARG );

};


} // Gtk
} // Falcon

#endif // !GTK_IMAGE_HPP
