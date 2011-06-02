#ifndef GDK_DISPLAY_HPP
#define GDK_DISPLAY_HPP

#include "modgtk.hpp"

#define GET_DISPLAY( item ) \
        (((Gdk::Display*) (item).asObjectSafe() )->getObject())


namespace Falcon {
namespace Gdk {

/**
 *  \class Falcon::Gdk::Display
 */
class Display
    :
    public Gtk::CoreGObject
{
public:

    Display( const Falcon::CoreClass*, const GdkDisplay* = 0 );

    static Falcon::CoreObject* factory( const Falcon::CoreClass*, void*, bool );

    static void modInit( Falcon::Module* );

    GdkDisplay* getObject() const { return (GdkDisplay*) m_obj; }

    static FALCON_FUNC signal_closed( VMARG );

    static void on_closed( GdkDisplay*, gboolean, gpointer );

    static FALCON_FUNC signal_opened( VMARG );

    static void on_opened( GdkDisplay*, gpointer );

    static FALCON_FUNC open( VMARG );

    static FALCON_FUNC get_default( VMARG );

    static FALCON_FUNC get_name( VMARG );

    static FALCON_FUNC get_n_screens( VMARG );

    static FALCON_FUNC get_screen( VMARG );

    static FALCON_FUNC get_default_screen( VMARG );

    static FALCON_FUNC pointer_ungrab( VMARG );

    static FALCON_FUNC keyboard_ungrab( VMARG );

    static FALCON_FUNC pointer_is_grabbed( VMARG );
#if 0 // todo
    static FALCON_FUNC device_is_grabbed( VMARG );
#endif
    static FALCON_FUNC beep( VMARG );

    static FALCON_FUNC sync( VMARG );

    static FALCON_FUNC flush( VMARG );

    static FALCON_FUNC close( VMARG );
#if 0 // todo
    static FALCON_FUNC list_devices( VMARG );

    static FALCON_FUNC get_event( VMARG );

    static FALCON_FUNC peek_event( VMARG );

    static FALCON_FUNC put_event( VMARG );

    static FALCON_FUNC add_client_message_filter( VMARG );

    static FALCON_FUNC set_double_click_time( VMARG );

    static FALCON_FUNC set_double_click_distance( VMARG );

    static FALCON_FUNC get_pointer( VMARG );

    static FALCON_FUNC get_window_at_pointer( VMARG );

    static FALCON_FUNC set_pointer_hooks( VMARG );

    static FALCON_FUNC warp_pointer( VMARG );

    static FALCON_FUNC supports_cursor_color( VMARG );

    static FALCON_FUNC supports_cursor_alpha( VMARG );

    static FALCON_FUNC get_default_cursor_size( VMARG );

    static FALCON_FUNC get_maximal_cursor_size( VMARG );

    static FALCON_FUNC get_default_group( VMARG );

    static FALCON_FUNC supports_selection_notification( VMARG );

    static FALCON_FUNC request_selection_notification( VMARG );

    static FALCON_FUNC supports_clipboard_persistence( VMARG );

    static FALCON_FUNC store_clipboard( VMARG );

    static FALCON_FUNC supports_shapes( VMARG );

    static FALCON_FUNC supports_input_shapes( VMARG );

    static FALCON_FUNC supports_composite( VMARG );
#endif

};


} // Gdk
} // Falcon

#endif // !GDK_DISPLAY_HPP

// vi: set ai et sw=4:
// kate: replace-tabs on; shift-width 4;
