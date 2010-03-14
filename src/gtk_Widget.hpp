#ifndef GTK_WIDGET_HPP
#define GTK_WIDGET_HPP

#include "modgtk.hpp"

#include <gtk/gtk.h>


namespace Falcon {
namespace Gtk {

/**
 *  \class Falcon::Gtk::Widget
 */
class Widget
    :
    public Falcon::CoreObject
{
public:

    Widget( const Falcon::CoreClass*, const GtkWidget* = 0 );

    ~Widget() {}

    Falcon::CoreObject* clone() const { return 0; }

    bool getProperty( const Falcon::String&, Falcon::Item& ) const;

    bool setProperty( const Falcon::String&, const Falcon::Item& );

    static Falcon::CoreObject* factory( const Falcon::CoreClass*, void*, bool );

    static void modInit( Falcon::Module* );

    /*
     *  Signals
     */

    static FALCON_FUNC signal_accel_closures_changed( VMARG );

    static void on_accel_closures_changed( GtkWidget*, gpointer );

    //static FALCON_FUNC signal_button_press_event( VMARG );

    //static gboolean on_button_press_event( GtkWidget*, GdkEventButton*, gpointer );

    //static FALCON_FUNC signal_button_release_event( VMARG );

    //static gboolean on_button_release_event( GtkWidget*, GdkEventButton*, gpointer );

    static FALCON_FUNC signal_can_activate_accel( VMARG );

    static gboolean on_can_activate_accel( GtkWidget*, guint, gpointer );

    //static FALCON_FUNC signal_child_notify( VMARG );

    //static gboolean on_child_notify( GtkWidget*, GParamSpec*, gpointer );

    //static FALCON_FUNC signal_client_event( VMARG );

    //static gboolean on_client_event( GtkWidget*, GdkEventClient*, gpointer );

    static FALCON_FUNC signal_composited_changed( VMARG );

    static void on_composited_changed( GtkWidget*, gpointer );

    //static FALCON_FUNC signal_configure_event( VMARG );

    //static gboolean on_configure_event( GtkWidget*, GdkEventConfigure*, gpointer );

    //static FALCON_FUNC signal_damage_event( VMARG );

    //static gboolean on_damage_event( GtkWidget*, GdkEvent*, gpointer );

    static FALCON_FUNC signal_delete_event( VMARG );

    static gboolean on_delete_event( GtkWidget*, GdkEvent*, gpointer );

    static FALCON_FUNC signal_destroy_event( VMARG );

    static gboolean on_destroy_event( GtkWidget*, GdkEvent*, gpointer );

    //static FALCON_FUNC signal_direction_changed( VMARG );

    //static void on_direction_changed( GtkWidget*, GtkTextDirection, gpointer );

    //static FALCON_FUNC signal_drag_begin( VMARG );

    //static void on_drag_begin( GtkWidget*, GdkDragContext*, gpointer );

    //static FALCON_FUNC signal_drag_data_delete( VMARG );

    //static void on_drag_data_delete( GtkWidget*, GdkDragContext*, gpointer );

    //static FALCON_FUNC signal_drag_data_get( VMARG );

    //static void on_drag_data_get( GtkWidget*, GdkDragContext*, GtkSelectionData*,
            //guint, guint, gpointer );

    //static FALCON_FUNC signal_drag_data_received( VMARG );

    //static void on_drag_data_received( GtkWidget*, GdkDragContext*, gint, gint,
            //GtkSelectionData*, guint, guint, gpointer );

    //static FALCON_FUNC signal_drag_drop( VMARG );

    //static gboolean on_drag_drop( GtkWidget*, GdkDragContext*, gint, gint, guint, gpointer );

    //static FALCON_FUNC signal_drag_end( VMARG );

    //static void on_drag_end( GtkWidget*, GdkDragContext*, gpointer );

    //static FALCON_FUNC signal_drag_failed( VMARG );

    //static gboolean on_drag_failed( GtkWidget*, GdkDragContext*, GtkDragResult, gpointer );

    //static FALCON_FUNC signal_drag_leave( VMARG );

    //static void on_drag_leave( GtkWidget*, GdkDragContext*, guint, gpointer );

    //static FALCON_FUNC signal_drag_motion( VMARG );

    //static gboolean on_drag_motion( GtkWidget*, GdkDragContext*, gint, gint, guint, gpointer );

    //static FALCON_FUNC signal_enter_notify_event( VMARG );

    //static gboolean on_enter_notify_event( GtkWidget*, GdkEventCrossing*, gpointer );

    //static FALCON_FUNC signal_event( VMARG );

    //static gboolean on_event( GtkWidget*, GdkEvent*, gpointer );

    //static FALCON_FUNC signal_event( VMARG );

    //static gboolean on_event( GtkWidget*, GdkEvent*, gpointer );

    //static FALCON_FUNC signal_event_after( VMARG );

    //static void on_event_after( GtkWidget*, GdkEvent*, gpointer );

    //static FALCON_FUNC signal_expose_event( VMARG );

    //static gboolean on_expose_event( GtkWidget*, GdkEventExpose*, gpointer );

    //static FALCON_FUNC signal_focus( VMARG );

    //static gboolean on_focus( GtkWidget*, GtkDirectionType*, gpointer );

    //static FALCON_FUNC signal_focus_in_event( VMARG );

    //static gboolean on_focus_in_event( GtkWidget*, GdkEventFocus*, gpointer );

    //static FALCON_FUNC signal_focus_out_event( VMARG );

    //static gboolean on_focus_out_event( GtkWidget*, GdkEventFocus*, gpointer );

    //static FALCON_FUNC signal_grab_broken_event( VMARG );

    //static gboolean on_grab_broken_event( GtkWidget*, GdkEvent*, gpointer );

    //static FALCON_FUNC signal_grab_focus( VMARG );

    //static void on_grab_focus( GtkWidget*, gpointer );

    //static FALCON_FUNC signal_grab_notify( VMARG );

    //static void on_grab_notify( GtkWidget*, gboolean, gpointer );

    static FALCON_FUNC signal_hide( VMARG );

    static void on_hide( GtkWidget*, gpointer );

    //static FALCON_FUNC signal_hierarchy_changed( VMARG );

    //static void on_hierarchy_changed( GtkWidget*, GtkWidget*, gpointer );

    //static FALCON_FUNC signal_key_press_event( VMARG );

    //static gboolean on_key_press_event( GtkWidget*, GdkEventKey*, gpointer );

    //static FALCON_FUNC signal_key_release_event( VMARG );

    //static gboolean on_key_release_event( GtkWidget*, GdkEventKey*, gpointer );

    //static FALCON_FUNC signal_keynav_failed( VMARG );

    //static gboolean on_keynav_failed( GtkWidget*, GtkDirectionType, gpointer );

    //static FALCON_FUNC signal_leave_notify_event( VMARG );

    //static gboolean on_leave_notify_event( GtkWidget*, GdkEventCrossing*, gpointer );

    //static FALCON_FUNC signal_map( VMARG );

    //static void on_map( GtkWidget*, gpointer );

    //static FALCON_FUNC signal_map_event( VMARG );

    //static gboolean on_map_event( GtkWidget*, GdkEvent*, gpointer );

    //static FALCON_FUNC signal_mnemonic_activate( VMARG );

    //static gboolean on_mnemonic_activate( GtkWidget*, gboolean, gpointer );

    //static FALCON_FUNC signal_motion_notify_event( VMARG );

    //static gboolean on_motion_notify_event( GtkWidget*, GdkEventMotion*, gpointer );

    //static FALCON_FUNC signal_move_focus( VMARG );

    //static void on_move_focus( GtkWidget*, GtkDirectionType, gpointer );

    //static FALCON_FUNC signal_no_expose_event( VMARG );

    //static gboolean on_no_expose_event( GtkWidget*, GdkEventNoExpose*, gpointer );

    //static FALCON_FUNC signal_parent_set( VMARG );

    //static void on_parent_set( GtkWidget*, GtkObject*, gpointer );

    //static FALCON_FUNC signal_popup_menu( VMARG );

    //static gboolean on_popup_menu( GtkWidget*, gpointer );

    //static FALCON_FUNC signal_property_notify_event( VMARG );

    //static gboolean on_property_notify_event( GtkWidget*, GdkEventProperty*, gpointer );

    //static FALCON_FUNC signal_proximity_in_event( VMARG );

    //static gboolean on_proximity_in_event( GtkWidget*, GdkEventProximity*, gpointer );

    //static FALCON_FUNC signal_proximity_out_event( VMARG );

    //static gboolean on_proximity_out_event( GtkWidget*, GdkEventProximity*, gpointer );

    //static FALCON_FUNC signal_query_tooltip( VMARG );

    //static gboolean on_query_tooltip( GtkWidget*, gint, gint, gboolean, GtkTooltip*, gpointer );

    //static FALCON_FUNC signal_realize( VMARG );

    //static void on_realize( GtkWidget*, gpointer );

    //static FALCON_FUNC signal_screen_changed( VMARG );

    //static void on_screen_changed( GtkWidget*, GdkScreen*, gpointer );

    //static FALCON_FUNC signal_scroll_event( VMARG );

    //static gboolean on_scroll_event( GtkWidget*, GdkEventScroll*, gpointer );

    //static FALCON_FUNC signal_selection_clear_event( VMARG );

    //static gboolean on_selection_clear_event( GtkWidget*, GdkEventSelection*, gpointer );

    //static FALCON_FUNC signal_selection_get( VMARG );

    //static void on_selection_get( GtkWidget*, GtkSelectionData*, guint, guint, gpointer );

    //static FALCON_FUNC signal_selection_notify_event( VMARG );

    //static gboolean on_selection_notify_event( GtkWidget*, GdkEventSelection*, gpointer );

    //static FALCON_FUNC signal_selection_received( VMARG );

    //static void on_selection_received( GtkWidget*, GtkSelectionData*, guint, gpointer );

    //static FALCON_FUNC signal_selection_request_event( VMARG );

    //static gboolean on_selection_request_event( GtkWidget*, GdkEventSelection*, gpointer );

    static FALCON_FUNC signal_show( VMARG );

    static void on_show( GtkWidget*, gpointer );

    //static FALCON_FUNC signal_show_help( VMARG );

    //static gboolean on_show_help( GtkWidget*, GtkWidgetHelpType, gpointer );

    //static FALCON_FUNC signal_size_allocate( VMARG );

    //static void on_size_allocate( GtkWidget*, GtkAllocation*, gpointer );

    //static FALCON_FUNC signal_size_request( VMARG );

    //static void on_size_request( GtkWidget*, GtkRequisition*, gpointer );

    //static FALCON_FUNC signal_state_changed( VMARG );

    //static void on_state_changed( GtkWidget*, GtkState, gpointer );

    //static FALCON_FUNC signal_style_set( VMARG );

    //static void on_style_set( GtkWidget*, GtkStyle*, gpointer );

    //static FALCON_FUNC signal_unmap( VMARG );

    //static void on_unmap( GtkWidget*, gpointer );

    //static FALCON_FUNC signal_unmap_event( VMARG );

    //static gboolean on_unmap_event( GtkWidget*, GdkEvent*, gpointer );

    //static FALCON_FUNC signal_unrealize( VMARG );

    //static void on_unrealize( GtkWidget*, gpointer );

    //static FALCON_FUNC signal_visibility_notify_event( VMARG );

    //static gboolean on_visibility_notify_event( GtkWidget*, GdkEventVisibility*, gpointer );

    //static FALCON_FUNC signal_window_state_event( VMARG );

    //static gboolean on_window_state_event( GtkWidget*, GdkEventWindowState*, gpointer );

    /*
     *  Methods
     */

    static FALCON_FUNC show( VMARG );

    static FALCON_FUNC show_now( VMARG );

    static FALCON_FUNC hide( VMARG );

    static FALCON_FUNC show_all( VMARG );

    static FALCON_FUNC hide_all( VMARG );

    static FALCON_FUNC activate( VMARG );

    static FALCON_FUNC reparent( VMARG );

    static FALCON_FUNC is_focus( VMARG );

    static FALCON_FUNC grab_focus( VMARG );

    static FALCON_FUNC grab_default( VMARG );

    static FALCON_FUNC set_name( VMARG );

    static FALCON_FUNC get_name( VMARG );

    static FALCON_FUNC set_sensitive( VMARG );

    static FALCON_FUNC get_toplevel( VMARG );

    static FALCON_FUNC get_events( VMARG );

    static FALCON_FUNC is_ancestor( VMARG );

    static FALCON_FUNC hide_on_delete( VMARG );

};

} // Gtk
} // Falcon

#endif // !GTK_WIDGET_HPP
