/**
 *  \file gtk_Widget.cpp
 */

#include "gtk_Widget.hpp"

#include "g_ParamSpec.hpp"

#include "gdk_Bitmap.hpp"
#include "gdk_Color.hpp"
#include "gdk_Colormap.hpp"
#include "gdk_Event.hpp"
#include "gdk_EventButton.hpp"
#include "gdk_Rectangle.hpp"
#include "gdk_Visual.hpp"
#include "gdk_Window.hpp"

#include "gtk_CellEditable.hpp"
//#include "gtk_ExtendedLayout.hpp"
#include "gtk_FileChooser.hpp"
#include "gtk_Requisition.hpp"
#include "gtk_ToolShell.hpp"


namespace Falcon {
namespace Gtk {


void Widget::modInit( Falcon::Module* mod )
{
    Falcon::Symbol* c_Widget = mod->addClass( "GtkWidget", &Gtk::abstract_init );

    Falcon::InheritDef* in = new Falcon::InheritDef( mod->findGlobalSymbol( "GtkObject" ) );
    c_Widget->getClassDef()->addInheritance( in );

    c_Widget->setWKS( true );
    c_Widget->getClassDef()->factory( &Widget::factory );

    Gtk::MethodTab methods[] =
    {
    { "signal_accel_closures_changed",  &Widget::signal_accel_closures_changed },
    { "signal_button_press_event",      &Widget::signal_button_press_event },
    { "signal_button_release_event",    &Widget::signal_button_release_event },
    { "signal_can_activate_accel",      &Widget::signal_can_activate_accel },
    { "signal_child_notify",            &Widget::signal_child_notify },
    //{ "signal_client_event",            &Widget::signal_client_event },
    { "signal_composited_changed",      &Widget::signal_composited_changed },
    //{ "signal_configure_event",         &Widget::signal_configure_event },
    //{ "signal_damage_event",            &Widget::signal_damage_event },
    { "signal_delete_event",            &Widget::signal_delete_event },
    { "signal_destroy_event",           &Widget::signal_destroy_event },
    //{ "signal_direction_changed",       &Widget::signal_direction_changed },
    //{ "signal_drag_begin",              &Widget::signal_drag_begin },
    //{ "signal_drag_data_delete",        &Widget::signal_drag_data_delete },
    //{ "signal_drag_data_get",           &Widget::signal_drag_data_get },
    //{ "signal_drag_data_received",      &Widget::signal_drag_data_received },
    //{ "signal_drag_drop",               &Widget::signal_drag_drop },
    //{ "signal_drag_end",                &Widget::signal_drag_end },
    //{ "signal_drag_failed",             &Widget::signal_drag_failed },
    //{ "signal_drag_leave",              &Widget::signal_drag_leave },
    //{ "signal_drag_motion",             &Widget::signal_drag_motion },
    //{ "signal_enter_notify_event",      &Widget::signal_enter_notify_event },
    //{ "signal_event",                   &Widget::signal_event },
    //{ "signal_event_after",             &Widget::signal_event_after },
    //{ "signal_expose_event",            &widget::signal_expose_event },
    //{ "signal_focus",                   &Widget::signal_focus },
    //{ "signal_focus_in_event",          &Widget::signal_focus_in_event },
    //{ "signal_focus_out_event",         &Widget::signal_focus_out_event },
    //{ "signal_grab_broken_event",       &Widget::signal_grab_broken_event },
    //{ "signal_grab_focus",              &Widget::signal_grab_focus },
    //{ "signal_grab_notify",             &Widget::signal_grab_notify },
    { "signal_hide",                    &Widget::signal_hide },
    //{ "signal_hierarchy_changed",       &Widget::signal_hierarchy_changed },
    //{ "signal_key_press_event",         &Widget::signal_key_press_event },
    //{ "signal_key_release_event",       &Widget::signal_key_release_event },
    //{ "signal_keynav_failed",           &Widget::signal_keynav_failed },
    //{ "signal_leave_notify_event",      &Widget::signal_leave_notify_event },
    //{ "signal_map",                     &Widget::signal_map },
    //{ "signal_map_event",               &Widget::signal_map_event },
    //{ "signal_mnemonic_activate",       &Widget::signal_mnemonic_activate },
    //{ "signal_motion_notify_event",     &Widget::signal_motion_notify_event },
    //{ "signal_move_focus",              &Widget::signal_move_focus },
    //{ "signal_no_expose_event",         &Widget::signal_no_expose_event },
    //{ "signal_parent_set",              &Widget::signal_parent_set },
    //{ "signal_popup_menu",              &Widget::signal_popup_menu },
    //{ "signal_property_notify_event",   &Widget::signal_property_notify_event },
    //{ "signal_proximity_in_event",      &Widget::signal_proximity_in_event },
    //{ "signal_proximity_out_event",     &Widget::signal_proximity_out_event },
    //{ "signal_query_tooltip",           &Widget::signal_query_tooltip },
    //{ "signal_realize",                 &Widget::signal_realize },
    //{ "signal_screen_changed",          &Widget::signal_screen_changed },
    //{ "signal_scroll_event",            &Widget::signal_scroll_event },
    //{ "signal_selection_clear_event",   &Widget::signal_selection_clear_event },
    //{ "signal_selection_get",           &Widget::signal_selection_get },
    //{ "signal_selection_notify_event",  &Widget::signal_selection_notify_event },
    //{ "signal_selection_received",      &Widget::signal_selection_received },
    //{ "signal_selection_request_event", &Widget::signal_selection_request_event },
    { "signal_show",                    &Widget::signal_show },
    //{ "signal_show_help",               &Widget::signal_show_help },
    //{ "signal_size_allocate",           &Widget::signal_size_allocate },
    { "signal_size_request",            &Widget::signal_size_request },
    //{ "signal_state_changed",           &Widget::signal_state_changed },
    //{ "signal_style_set",               &Widget::signal_style_set },
    //{ "signal_unmap",                   &Widget::signal_unmap },
    //{ "signal_unmap_event",             &Widget::signal_unmap_event },
    //{ "signal_unrealize",               &Widget::signal_unrealize },
    //{ "signal_visibility_notify_event", &Widget::signal_visibility_notify_event },
    //{ "signal_window_state_event",      &Widget::signal_window_state_event },
    { "destroy",                &Widget::destroy },
#if 0 // unused
    { "destroyed",              &Widget::destroyed },
#endif
    { "unparent",               &Widget::unparent },
    { "show",                   &Widget::show },
    { "show_now",               &Widget::show_now },
    { "hide",                   &Widget::hide },
    { "show_all",               &Widget::show_all },
    { "hide_all",               &Widget::hide_all },
    { "map",                    &Widget::map },
    { "unmap",                  &Widget::unmap },
    { "realize",                &Widget::realize },
    { "unrealize",              &Widget::unrealize },
    { "queue_draw",             &Widget::queue_draw },
    { "queue_resize",           &Widget::queue_resize },
    { "queue_resize_no_redraw", &Widget::queue_resize_no_redraw },
    { "size_request",           &Widget::size_request },
    { "get_child_requisition",  &Widget::get_child_requisition },
#if 0 // todo
    { "size_allocate",          &Widget::size_allocate },
    { "add_accelerator",        &Widget::add_accelerator },
    { "remove_accelerator",     &Widget::remove_accelerator },
    { "set_accel_path",         &Widget::set_accel_path },
    { "list_accel_closures",    &Widget::list_accel_closures },
#endif
    { "can_activate_accel",     &Widget::can_activate_accel },
    { "event",                  &Widget::event },
    { "activate",               &Widget::activate },
    { "reparent",               &Widget::reparent },
    { "intersect",              &Widget::intersect },
    { "is_focus",               &Widget::is_focus },
    { "grab_focus",             &Widget::grab_focus },
    { "grab_default",           &Widget::grab_default },
    { "set_name",               &Widget::set_name },
    { "get_name",               &Widget::get_name },
    { "set_state",              &Widget::set_state },
    { "set_sensitive",          &Widget::set_sensitive },
    { "set_parent",             &Widget::set_parent },
    { "set_parent_window",      &Widget::set_parent_window },
    { "get_parent_window",      &Widget::get_parent_window },
    { "set_events",             &Widget::set_events },
    { "get_events",             &Widget::get_events },
    { "add_events",             &Widget::add_events },
    { "set_extension_events",   &Widget::set_extension_events },
    { "get_extension_events",   &Widget::get_extension_events },
#if GTK_CHECK_VERSION( 3, 0, 0 )
    { "set_device_events",      &Widget::set_device_events },
    { "get_device_events",      &Widget::get_device_events },
    { "add_device_events",      &Widget::add_device_events },
#endif
    { "get_toplevel",           &Widget::get_toplevel },
    { "get_ancestor",           &Widget::get_ancestor },
    { "get_colormap",           &Widget::get_colormap },
    { "set_colormap",           &Widget::set_colormap },
    { "get_visual",             &Widget::get_visual },
    { "get_pointer",            &Widget::get_pointer },
    { "is_ancestor",            &Widget::is_ancestor },
    { "translate_coordinates",  &Widget::translate_coordinates },
    { "hide_on_delete",         &Widget::hide_on_delete },
#if 0 // todo
    { "set_style",              &Widget::set_style },
#endif
    { "ensure_style",           &Widget::ensure_style },
#if 0 // todo
    { "get_style",              &Widget::get_style },
#endif
    { "reset_rc_styles",        &Widget::reset_rc_styles },
    { "push_colormap",          &Widget::push_colormap },
    { "pop_colormap",           &Widget::pop_colormap },
    { "set_default_colormap",   &Widget::set_default_colormap },
#if 0 // todo
    { "get_default_style",      &Widget::get_default_style },
#endif
    { "get_default_colormap",   &Widget::get_default_colormap },
    { "get_default_visual",     &Widget::get_default_visual },
    { "set_direction",          &Widget::set_direction },
    { "get_direction",          &Widget::get_direction },
    { "set_default_direction",  &Widget::set_default_direction },
    { "get_default_direction",  &Widget::get_default_direction },
    { "shape_combine_mask",     &Widget::shape_combine_mask },
    { "input_shape_combine_mask",&Widget::input_shape_combine_mask },
    { "path",                   &Widget::path },
    { "class_path",             &Widget::class_path },
    { "get_composite_name",     &Widget::get_composite_name },
#if 0 // todo
    { "modify_style",    &Widget:: },
    { "get_modifier_style",    &Widget:: },
#endif
    { "modify_fg",              &Widget::modify_fg },
    { "modify_bg",              &Widget::modify_bg },
    { "modify_text",            &Widget::modify_text },
    { "modify_base",            &Widget::modify_base },
#if 0
    { "modify_font",    &Widget:: },
    { "modify_cursor",    &Widget:: },
    { "create_pango_context",    &Widget:: },
    { "get_pango_context",    &Widget:: },
    { "create_pango_layout",    &Widget:: },
    { "widget_render_icon",    &Widget:: },
    { "pop_composite_child",    &Widget:: },
    { "push_composite_child",    &Widget:: },
    { "queue_clear",    &Widget:: },
    { "queue_clear_area",    &Widget:: },
    { "queue_draw_area",    &Widget:: },
    { "reset_shapes",    &Widget:: },
    { "set_app_paintable",    &Widget:: },
    { "set_double_buffered",    &Widget:: },
    { "set_redraw_on_allocate",    &Widget:: },
    { "set_composite_name",    &Widget:: },
    { "set_scroll_adjustments",    &Widget:: },
    { "mnemonic_activate",    &Widget:: },
    { "class_install_style_property",    &Widget:: },
    { "class_install_style_property_parser",    &Widget:: },
    { "class_find_style_property",    &Widget:: },
    { "class_list_style_properties",    &Widget:: },
    { "region_intersect",    &Widget:: },
    { "send_expose",    &Widget:: },
    { "style_get",    &Widget:: },
    { "style_get_property",    &Widget:: },
    { "style_get_valist",    &Widget:: },
    { "style_attach",    &Widget:: },
    { "get_accessible",    &Widget:: },
    { "child_focus",    &Widget:: },
    { "child_notify",    &Widget:: },
    { "freeze_child_notify",    &Widget:: },
    { "get_child_visible",    &Widget:: },
    { "get_parent",    &Widget:: },
    { "get_settings",    &Widget:: },
    { "get_clipboard",    &Widget:: },
    { "get_display",    &Widget:: },
    { "get_root_window",    &Widget:: },
    { "get_screen",    &Widget:: },
    { "has_screen",    &Widget:: },
#endif


    { "get_size_request",       &Widget::get_size_request },
    //{ "set_child_visible",      &Widget::set_child_visible },
    //{ "set_default_visual",     &Widget::set_default_visual },
    { "set_size_request",       &Widget::set_size_request },

    { NULL, NULL }
    };

    for ( Gtk::MethodTab* meth = methods; meth->name; ++meth )
        mod->addClassMethod( c_Widget, meth->name, meth->cb );

    Gtk::CellEditable::clsInit( mod, c_Widget );
    //Gtk::ExtendedLayout::clsInit( mod, c_Widget );
    Gtk::FileChooser::clsInit( mod, c_Widget );
    Gtk::ToolShell::clsInit( mod, c_Widget );
}


Widget::Widget( const Falcon::CoreClass* gen, const GtkWidget* wdt )
    :
    Gtk::CoreGObject( gen, (GObject*) wdt )
{}


Falcon::CoreObject* Widget::factory( const Falcon::CoreClass* gen, void* wdt, bool )
{
    return new Widget( gen, (GtkWidget*) wdt );
}


/*#
    @class GtkWidget
    @brief Base class for all widgets

    GtkWidget is the base class all widgets in GTK+ derive from.
    It manages the widget lifecycle, states and style.

    [...]
 */


/*#
    @method signal_accel_closures_changed GtkWidget
    @brief Connect a VMSlot to the widget accel_closures_changed signal and return it
 */
FALCON_FUNC Widget::signal_accel_closures_changed( VMARG )
{
    NO_ARGS
    CoreGObject::get_signal( "accel_closures_changed",
        (void*) &Widget::on_accel_closures_changed, vm );
}


void Widget::on_accel_closures_changed( GtkWidget* wdt, gpointer _vm )
{
    CoreGObject::trigger_slot( (GObject*) wdt, "accel_closures_changed",
        "on_accel_closures_changed", (VMachine*)_vm );
}


/*#
    @method signal_button_press_event GtkWidget
    @brief Connect a VMSlot to the widget button_press_event signal and return it

    The button-press-event signal will be emitted when a button (typically from a mouse)
    is pressed.

    To receive this signal, the GdkWindow associated to the widget needs to enable
    the GDK_BUTTON_PRESS_MASK mask.

    This signal will be sent to the grab widget if there is one.

    Your callback function gets a GdkEventButton as parameter, and must return
    a boolean, true to stop other handlers from being invoked for the event,
    false to propagate the event further.
 */
FALCON_FUNC Widget::signal_button_press_event( VMARG )
{
    NO_ARGS
    CoreGObject::get_signal( "button_press_event",
        (void*) &Widget::on_button_press_event, vm );
}


gboolean Widget::on_button_press_event( GtkWidget* obj, GdkEventButton* ev, gpointer _vm )
{
    GET_SIGNALS( obj );
    CoreSlot* cs = _signals->getChild( "button_press_event", false );

    if ( !cs || cs->empty() )
        return FALSE; // propagate event

    VMachine* vm = (VMachine*) _vm;
    Iterator iter( cs );
    Item it;
    Item* wki = vm->findWKI( "GdkEventButton" );

    do
    {
        it = iter.getCurrent();

        if ( !it.isCallable() )
        {
            if ( !it.isComposed()
                || !it.asObject()->getMethod( "on_button_press_event", it ) )
            {
                printf(
                "[GtkWidget::on_button_press_event] invalid callback (expected callable)\n" );
                return TRUE; // block event
            }
        }
        vm->pushParam( new Gdk::EventButton( wki->asClass(), ev ) );
        vm->callItem( it, 1 );
        it = vm->regA();

        if ( !it.isNil() && it.isBoolean() )
        {
            if ( it.asBoolean() )
                return TRUE; // block event
            else
                iter.next();
        }
        else
        {
            printf(
            "[GtkWidget::on_button_press_event] invalid callback (expected boolean)\n" );
            return TRUE; // block event
        }
    }
    while ( iter.hasCurrent() );

    return FALSE; // propagate event
}


/*#
    @method signal_button_release_event GtkWidget
    @brief Connect a VMSlot to the widget button_release_event signal and return it

    The button-release-event signal will be emitted when a button (typically from a mouse)
    is released.

    To receive this signal, the GdkWindow associated to the widget needs to enable
    the GDK_BUTTON_RELEASE_MASK mask.

    This signal will be sent to the grab widget if there is one.
 */
FALCON_FUNC Widget::signal_button_release_event( VMARG )
{
    NO_ARGS
    CoreGObject::get_signal( "button_release_event",
        (void*) &Widget::on_button_release_event, vm );
}


gboolean Widget::on_button_release_event( GtkWidget* obj, GdkEventButton* ev, gpointer _vm )
{
    GET_SIGNALS( obj );
    CoreSlot* cs = _signals->getChild( "button_release_event", false );

    if ( !cs || cs->empty() )
        return FALSE; // propagate event

    VMachine* vm = (VMachine*) _vm;
    Iterator iter( cs );
    Item it;
    Item* wki = vm->findWKI( "GdkEventButton" );

    do
    {
        it = iter.getCurrent();

        if ( !it.isCallable() )
        {
            if ( !it.isComposed()
                || !it.asObject()->getMethod( "on_button_release_event", it ) )
            {
                printf(
                "[GtkWidget::on_button_release_event] invalid callback (expected callable)\n" );
                return TRUE; // block event
            }
        }
        vm->pushParam( new Gdk::EventButton( wki->asClass(), ev ) );
        vm->callItem( it, 1 );
        it = vm->regA();

        if ( !it.isNil() && it.isBoolean() )
        {
            if ( it.asBoolean() )
                return TRUE; // block event
            else
                iter.next();
        }
        else
        {
            printf(
            "[GtkWidget::on_button_release_event] invalid callback (expected boolean)\n" );
            return TRUE; // block event
        }
    }
    while ( iter.hasCurrent() );

    return FALSE; // propagate event
}


/*#
    @method signal_can_activate_accel GtkWidget
    @brief Connect a VMSlot to the widget can_activate_accel signal and return it

    Determines whether an accelerator that activates the signal identified by
    signal_id can currently be activated. This signal is present to allow applications
    and derived widgets to override the default GtkWidget handling for determining
    whether an accelerator can be activated.

    The callback function must return a boolean (returning true if the signal can
    be activated). It will get an integer as parameter, that is the ID of a signal
    installed on widget.
 */
FALCON_FUNC Widget::signal_can_activate_accel( VMARG )
{
    NO_ARGS
    CoreGObject::get_signal( "can_activate_accel",
        (void*) &Widget::on_can_activate_accel, vm );
}


gboolean Widget::on_can_activate_accel( GtkWidget* obj, guint signal_id, gpointer _vm )
{
    GET_SIGNALS( obj );
    CoreSlot* cs = _signals->getChild( "can_activate_accel", false );

    if ( !cs || cs->empty() )
        return TRUE; // can activate

    VMachine* vm = (VMachine*) _vm;
    Iterator iter( cs );
    Item it;

    do
    {
        it = iter.getCurrent();

        if ( !it.isCallable() )
        {
            if ( !it.isComposed()
                || !it.asObject()->getMethod( "on_can_activate_accel", it ) )
            {
                printf(
                "[GtkWidget::on_can_activate_accel] invalid callback (expected callable)\n" );
                return FALSE; // block event
            }
        }
        vm->pushParam( (int64) signal_id );
        vm->callItem( it, 1 );
        it = vm->regA();

        if ( !it.isNil() && it.isBoolean() )
        {
            if ( !it.asBoolean() )
                return FALSE; // block event
            else
                iter.next();
        }
        else
        {
            printf(
            "[GtkWidget::on_can_activate_accel] invalid callback (expected boolean)\n" );
            return FALSE; // block event
        }
    }
    while ( iter.hasCurrent() );

    return TRUE; // can activate
}


/*#
    @method signal_child_notify GtkWidget
    @brief Connect a VMSlot to the widget child_notify signal and return it

    The child-notify signal is emitted for each child property that has changed
    on an object. The signal's detail holds the property name.
 */
FALCON_FUNC Widget::signal_child_notify( VMARG )
{
    NO_ARGS
    CoreGObject::get_signal( "child_notify",
        (void*) &Widget::on_child_notify, vm );
}


void Widget::on_child_notify( GtkWidget* obj, GParamSpec* spec, gpointer _vm )
{
    GET_SIGNALS( obj );
    CoreSlot* cs = _signals->getChild( "child_notify", false );

    if ( !cs || cs->empty() )
        return;

    VMachine* vm = (VMachine*) _vm;
    Iterator iter( cs );
    Item it;
    Item* wki = vm->findWKI( "GParamSpec" );

    do
    {
        it = iter.getCurrent();

        if ( !it.isCallable() )
        {
            if ( !it.isComposed()
                || !it.asObject()->getMethod( "on_child_notify", it ) )
            {
                printf(
                "[GtkWidget::on_child_notify] invalid callback (expected callable)\n" );
                return;
            }
        }
        vm->pushParam( new Glib::ParamSpec( wki->asClass(), spec ) );
        vm->callItem( it, 1 );
    }
    while ( iter.hasCurrent() );
}


//FALCON_FUNC Widget::signal_client_event( VMARG );

//gboolean Widget::on_client_event( GtkWidget*, GdkEventClient*, gpointer );


/*#
    @method signal_composited_changed GtkWidget
    @brief Connect a VMSlot to the widget composited_changed signal and return it

    The composited-changed signal is emitted when the composited status of
    widgets screen changes.
 */
FALCON_FUNC Widget::signal_composited_changed( VMARG )
{
    NO_ARGS
    CoreGObject::get_signal( "composited_changed",
        (void*) &Widget::on_composited_changed, vm );
}


void Widget::on_composited_changed( GtkWidget* obj, gpointer _vm )
{
    CoreGObject::trigger_slot( (GObject*) obj, "composited_changed",
        "on_composited_changed", (VMachine*)_vm );
}


//FALCON_FUNC Widget::signal_configure_event( VMARG );

//gboolean Widget::on_configure_event( GtkWidget*, GdkEventConfigure*, gpointer );

//FALCON_FUNC signal_damage_event( VMARG );

//gboolean on_damage_event( GtkWidget*, GdkEvent*, gpointer );


/*#
    @method signal_delete_event GtkWidget
    @brief Connect a VMSlot to the widget delete_event signal and return it

    The callback function must return a boolean, that if is true, will block the event.

    The delete-event signal is emitted if a user requests that a toplevel window
    is closed. The default handler for this signal destroys the window.
    Connecting gtk_widget_hide_on_delete() to this signal will cause the window
    to be hidden instead, so that it can later be shown again without reconstructing it.
 */
FALCON_FUNC Widget::signal_delete_event( VMARG )
{
    NO_ARGS
    CoreGObject::get_signal( "delete_event", (void*) &Widget::on_delete_event, vm );
}


gboolean Widget::on_delete_event( GtkWidget* obj, GdkEvent* ev, gpointer _vm )
{
    GET_SIGNALS( obj );
    CoreSlot* cs = _signals->getChild( "delete_event", false );

    if ( !cs || cs->empty() )
        return FALSE; // propagate event

    VMachine* vm = (VMachine*) _vm;
    Iterator iter( cs );
    Item it;

    do
    {
        it = iter.getCurrent();

        if ( !it.isCallable() )
        {
            if ( !it.isComposed()
                || !it.asObject()->getMethod( "on_delete_event", it ) )
            {
                printf(
                "[GtkWidget::on_delete_event] invalid callback (expected callable)\n" );
                return TRUE; // block event
            }
        }
        //vm->pushParam( Item( (int64)((GdkEventAny*)ev)->type ) );
        vm->callItem( it, 0 );
        it = vm->regA();

        if ( !it.isNil() && it.isBoolean() )
        {
            if ( it.asBoolean() )
                return TRUE; // block event
            else
                iter.next();
        }
        else
        {
            printf(
            "[GtkWidget::on_delete_event] invalid callback (expected boolean)\n" );
            return TRUE; // block event
        }
    }
    while ( iter.hasCurrent() );

    return FALSE; // propagate event
}


/*#
    @method signal_destroy_event GtkWidget
    @brief Connect a VMSlot to the widget destroy signal and return it

    The callback function must return a boolean, true to stop other handlers from
    being invoked for the event, false to propagate the event further.

    The destroy-event signal is emitted when a GdkWindow is destroyed.
    You rarely get this signal, because most widgets disconnect themselves from
    their window before they destroy it, so no widget owns the window at destroy time.

    To receive this signal, the GdkWindow associated to the widget needs to enable
    the GDK_STRUCTURE_MASK mask. GDK will enable this mask automatically for all new windows.
 */
FALCON_FUNC Widget::signal_destroy_event( VMARG )
{
    NO_ARGS
    CoreGObject::get_signal( "destroy_event", (void*) &Widget::on_destroy_event, vm );
}


gboolean Widget::on_destroy_event( GtkWidget* obj, GdkEvent*, gpointer _vm )
{
    GET_SIGNALS( obj );
    CoreSlot* cs = _signals->getChild( "destroy_event", false );

    if ( !cs || cs->empty() )
        return FALSE; // propagate event

    VMachine* vm = (VMachine*) _vm;
    Iterator iter( cs );
    Item it;

    do
    {
        it = iter.getCurrent();

        if ( !it.isCallable() )
        {
            if ( !it.isComposed()
                || !it.asObject()->getMethod( "on_destroy_event", it ) )
            {
                printf(
                "[GtkWidget::on_destroy_event] invalid callback (expected callable)\n" );
                return TRUE; // block event
            }
        }
        //vm->pushParam( Item( (int64)((GdkEventAny*)ev)->type ) );
        vm->callItem( it, 0 );
        it = vm->regA();

        if ( !it.isNil() && it.isBoolean() )
        {
            if ( it.asBoolean() )
                return TRUE; // block event
            else
                iter.next();
        }
        else
        {
            printf(
            "[GtkWidget::on_destroy_event] invalid callback (expected boolean)\n" );
            return TRUE; // block event
        }
    }
    while ( iter.hasCurrent() );

    return FALSE; // propagate event
}


//FALCON_FUNC Widget::signal_direction_changed( VMARG );

//void Widget::on_direction_changed( GtkWidget*, GtkTextDirection, gpointer );

//FALCON_FUNC Widget::signal_drag_begin( VMARG );

//void Widget::on_drag_begin( GtkWidget*, GdkDragContext*, gpointer );

//FALCON_FUNC Widget::signal_drag_data_delete( VMARG );

//void Widget::on_drag_data_delete( GtkWidget*, GdkDragContext*, gpointer );

//FALCON_FUNC Widget::signal_drag_data_get( VMARG );

//void Widget::on_drag_data_get( GtkWidget*, GdkDragContext*, GtkSelectionData*, guint, guint, gpointer );

//FALCON_FUNC Widget::signal_drag_data_received( VMARG );

//void Widget::on_drag_data_received( GtkWidget*, GdkDragContext*, gint, gint,
        //GtkSelectionData*, guint, guint, gpointer );

//FALCON_FUNC Widget::signal_drag_drop( VMARG );

//gboolean Widget::on_drag_drop( GtkWidget*, GdkDragContext*, gint, gint, guint, gpointer );

//FALCON_FUNC Widget::signal_drag_end( VMARG );

//void Widget::on_drag_end( GtkWidget*, GdkDragContext*, gpointer );

//FALCON_FUNC Widget::signal_drag_failed( VMARG );

//gboolean Widget::on_drag_failed( GtkWidget*, GdkDragContext*, GtkDragResult, gpointer );

//FALCON_FUNC Widget::signal_drag_leave( VMARG );

//void Widget::on_drag_leave( GtkWidget*, GdkDragContext*, guint, gpointer );

//FALCON_FUNC Widget::signal_drag_motion( VMARG );

//gboolean Widget::on_drag_motion( GtkWidget*, GdkDragContext*, gint, gint, guint, gpointer );

//FALCON_FUNC Widget::signal_enter_notify_event( VMARG );

//gboolean Widget::on_enter_notify_event( GtkWidget*, GdkEventCrossing*, gpointer );

//FALCON_FUNC Widget::signal_event( VMARG );

//gboolean Widget::on_event( GtkWidget*, GdkEvent*, gpointer );

//FALCON_FUNC Widget::signal_event_after( VMARG );

//void Widget::on_event_after( GtkWidget*, GdkEvent*, gpointer );

//FALCON_FUNC Widget::signal_expose_event( VMARG );

//gboolean Widget::on_expose_event( GtkWidget*, GdkEventExpose*, gpointer );

//FALCON_FUNC Widget::signal_focus( VMARG );

//gboolean Widget::on_focus( GtkWidget*, GtkDirectionType*, gpointer );

//FALCON_FUNC Widget::signal_focus_in_event( VMARG );

//gboolean Widget::on_focus_in_event( GtkWidget*, GdkEventFocus*, gpointer );

//FALCON_FUNC Widget::signal_focus_out_event( VMARG );

//gboolean Widget::on_focus_out_event( GtkWidget*, GdkEventFocus*, gpointer );

//FALCON_FUNC Widget::signal_grab_broken_event( VMARG );

//gboolean Widget::on_grab_broken_event( GtkWidget*, GdkEvent*, gpointer );

//FALCON_FUNC Widget::signal_grab_focus( VMARG );

//void Widget::on_grab_focus( GtkWidget*, gpointer );

//FALCON_FUNC Widget::signal_grab_notify( VMARG );

//void Widget::on_grab_notify( GtkWidget*, gboolean, gpointer );


/*#
    @method signal_hide GtkWidget
    @brief Connect a VMSlot to the widget hide signal and return it
 */
FALCON_FUNC Widget::signal_hide( VMARG )
{
    NO_ARGS
    CoreGObject::get_signal( "hide", (void*) &Widget::on_hide, vm );
}


void Widget::on_hide( GtkWidget* obj, gpointer _vm )
{
    CoreGObject::trigger_slot( (GObject*) obj, "hide", "on_hide", (VMachine*)_vm );
}




//FALCON_FUNC Widget::signal_hierarchy_changed( VMARG );

//void Widget::on_hierarchy_changed( GtkWidget*, GtkWidget*, gpointer );

//FALCON_FUNC Widget::signal_key_press_event( VMARG );

//gboolean Widget::on_key_press_event( GtkWidget*, GdkEventKey*, gpointer );

//FALCON_FUNC Widget::signal_key_release_event( VMARG );

//gboolean Widget::on_key_release_event( GtkWidget*, GdkEventKey*, gpointer );

//FALCON_FUNC Widget::signal_keynav_failed( VMARG );

//gboolean Widget::on_keynav_failed( GtkWidget*, GtkDirectionType, gpointer );

//FALCON_FUNC Widget::signal_leave_notify_event( VMARG );

//gboolean Widget::on_leave_notify_event( GtkWidget*, GdkEventCrossing*, gpointer );

//FALCON_FUNC Widget::signal_map( VMARG );

//void Widget::on_map( GtkWidget*, gpointer );

//FALCON_FUNC Widget::signal_map_event( VMARG );

//gboolean Widget::on_map_event( GtkWidget*, GdkEvent*, gpointer );

//FALCON_FUNC Widget::signal_mnemonic_activate( VMARG );

//gboolean Widget::on_mnemonic_activate( GtkWidget*, gboolean, gpointer );

//FALCON_FUNC Widget::signal_motion_notify_event( VMARG );

//gboolean Widget::on_motion_notify_event( GtkWidget*, GdkEventMotion*, gpointer );

//FALCON_FUNC Widget::signal_move_focus( VMARG );

//void Widget::on_move_focus( GtkWidget*, GtkDirectionType, gpointer );

//FALCON_FUNC Widget::signal_no_expose_event( VMARG );

//gboolean Widget::on_no_expose_event( GtkWidget*, GdkEventNoExpose*, gpointer );

//FALCON_FUNC Widget::signal_parent_set( VMARG );

//void Widget::on_parent_set( GtkWidget*, GtkObject*, gpointer );

//FALCON_FUNC Widget::signal_popup_menu( VMARG );

//gboolean Widget::on_popup_menu( GtkWidget*, gpointer );

//FALCON_FUNC Widget::signal_property_notify_event( VMARG );

//gboolean Widget::on_property_notify_event( GtkWidget*, GdkEventProperty*, gpointer );

//FALCON_FUNC Widget::signal_proximity_in_event( VMARG );

//gboolean Widget::on_proximity_in_event( GtkWidget*, GdkEventProximity*, gpointer );

//FALCON_FUNC Widget::signal_proximity_out_event( VMARG );

//gboolean Widget::on_proximity_out_event( GtkWidget*, GdkEventProximity*, gpointer );

//FALCON_FUNC Widget::signal_query_tooltip( VMARG );

//gboolean Widget::on_query_tooltip( GtkWidget*, gint, gint, gboolean, GtkTooltip*, gpointer );

//FALCON_FUNC Widget::signal_realize( VMARG );

//void Widget::on_realize( GtkWidget*, gpointer );

//FALCON_FUNC Widget::signal_screen_changed( VMARG );

//void Widget::on_screen_changed( GtkWidget*, GdkScreen*, gpointer );

//FALCON_FUNC Widget::signal_scroll_event( VMARG );

//gboolean Widget::on_scroll_event( GtkWidget*, GdkEventScroll*, gpointer );

//FALCON_FUNC Widget::signal_selection_clear_event( VMARG );

//gboolean Widget::on_selection_clear_event( GtkWidget*, GdkEventSelection*, gpointer );

//FALCON_FUNC Widget::signal_selection_get( VMARG );

//void Widget::on_selection_get( GtkWidget*, GtkSelectionData*, guint, guint, gpointer );

//FALCON_FUNC Widget::signal_selection_notify_event( VMARG );

//gboolean Widget::on_selection_notify_event( GtkWidget*, GdkEventSelection*, gpointer );

//FALCON_FUNC Widget::signal_selection_received( VMARG );

//void Widget::on_selection_received( GtkWidget*, GtkSelectionData*, guint, gpointer );

//FALCON_FUNC Widget::signal_selection_request_event( VMARG );

//gboolean Widget::on_selection_request_event( GtkWidget*, GdkEventSelection*, gpointer );


/*#
    @method signal_show GtkWidget
    @brief Connect a VMSlot to the widget show signal and return it
 */
FALCON_FUNC Widget::signal_show( VMARG )
{
    NO_ARGS
    CoreGObject::get_signal( "show", (void*) &Widget::on_show, vm );
}


void Widget::on_show( GtkWidget* obj, gpointer _vm )
{
    CoreGObject::trigger_slot( (GObject*) obj, "show", "on_show", (VMachine*)_vm );
}


//FALCON_FUNC Widget::signal_show_help( VMARG );

//gboolean Widget::on_show_help( GtkWidget*, GtkWidgetHelpType, gpointer );

//FALCON_FUNC Widget::signal_size_allocate( VMARG );

//void Widget::on_size_allocate( GtkWidget*, GtkAllocation*, gpointer );


/*#
    @method signal_size_request GtkWidget
    @brief Connect a VMSlot to the widget size-request signal and return it
 */
FALCON_FUNC Widget::signal_size_request( VMARG )
{
    NO_ARGS
    CoreGObject::get_signal( "size_request", (void*) &Widget::on_size_request, vm );
}


void Widget::on_size_request( GtkWidget* obj, GtkRequisition* req, gpointer _vm )
{
    GET_SIGNALS( obj );
    CoreSlot* cs = _signals->getChild( "size_request", false );

    if ( !cs || cs->empty() )
        return;

    VMachine* vm = (VMachine*)_vm;
    Item* wki = vm->findWKI( "GtkRequisition" );
    Iterator iter( cs );
    Item it;

    do
    {
        it = iter.getCurrent();
        if ( !it.isCallable() )
        {
            if ( !it.isComposed()
                || !it.asObject()->getMethod( "on_size_request", it ) )
            {
                printf(
                "[GtkWidget::on_size_request] invalid callback (expected callable)\n" );
                return;
            }
        }
        vm->pushParam( new Gtk::Requisition( wki->asClass(), req ) );
        vm->callItem( it, 1 );
        iter.next();
    }
    while ( iter.hasCurrent() );
}


//FALCON_FUNC Widget::signal_state_changed( VMARG );

//void Widget::on_state_changed( GtkWidget*, GtkState, gpointer );

//FALCON_FUNC Widget::signal_style_set( VMARG );

//void Widget::on_style_set( GtkWidget*, GtkStyle*, gpointer );

//FALCON_FUNC Widget::signal_unmap( VMARG );

//void Widget::on_unmap( GtkWidget*, gpointer );

//FALCON_FUNC Widget::signal_unmap_event( VMARG );

//gboolean Widget::on_unmap_event( GtkWidget*, GdkEvent*, gpointer );

//FALCON_FUNC Widget::signal_unrealize( VMARG );

//void Widget::on_unrealize( GtkWidget*, gpointer );

//FALCON_FUNC Widget::signal_visibility_notify_event( VMARG );

//gboolean Widget::on_visibility_notify_event( GtkWidget*, GdkEventVisibility*, gpointer );

//FALCON_FUNC Widget::signal_window_state_event( VMARG );

//gboolean Widget::on_window_state_event( GtkWidget*, GdkEventWindowState*, gpointer );



/*#
    @method destroy GtkWidget
    @brief Destroys a widget.

    Equivalent to gtk_object_destroy(), except that you don't have to cast the
    widget to GtkObject. When a widget is destroyed, it will break any references
    it holds to other objects. If the widget is inside a container, the widget
    will be removed from the container. If the widget is a toplevel (derived from
    GtkWindow), it will be removed from the list of toplevels, and the reference
    GTK+ holds to it will be removed. Removing a widget from its container or the
    list of toplevels results in the widget being finalized, unless you've added
    additional references to the widget with g_object_ref().

    In most cases, only toplevel widgets (windows) require explicit destruction,
    because when you destroy a toplevel its children will be destroyed as well.
 */
FALCON_FUNC Widget::destroy( VMARG )
{
    NO_ARGS
    gtk_widget_destroy( GET_WIDGET( vm->self() ) );
}


#if 0 // unused
FALCON_FUNC Widget::destroyed( VMARG );
#endif


/*#
    @method unparent GtkWidget
    @brief This function is only for use in widget implementations.

    Should be called by implementations of the remove method on GtkContainer,
    to dissociate a child from the container.
 */
FALCON_FUNC Widget::unparent( VMARG )
{
    NO_ARGS
    gtk_widget_unparent( GET_WIDGET( vm->self() ) );
}


/*#
    @method show GtkWidget
    @brief Flags a widget to be displayed.

    Any widget that isn't shown will not appear on the screen. If you want to show
    all the widgets in a container, it's easier to call gtk_widget_show_all() on
    the container, instead of individually showing the widgets.

    Remember that you have to show the containers containing a widget, in addition
    to the widget itself, before it will appear onscreen.

    When a toplevel container is shown, it is immediately realized and mapped;
    other shown widgets are realized and mapped when their toplevel container is
    realized and mapped.
 */
FALCON_FUNC Widget::show( VMARG )
{
    NO_ARGS
    gtk_widget_show( GET_WIDGET( vm->self() ) );
}


/*#
    @method show_now GtkWidget
    @brief Shows a widget.

    If the widget is an unmapped toplevel widget (i.e. a GtkWindow that has not
    yet been shown), enter the main loop and wait for the window to actually be
    mapped. Be careful; because the main loop is running, anything can happen
    during this function.
 */
FALCON_FUNC Widget::show_now( VMARG )
{
    NO_ARGS
    gtk_widget_show_now( GET_WIDGET( vm->self() ) );
}


/*#
    @method hide GtkWidget
    @brief Reverses the effects of show(), causing the widget to be hidden (invisible to the user).
 */
FALCON_FUNC Widget::hide( VMARG )
{
    NO_ARGS
    gtk_widget_hide( GET_WIDGET( vm->self() ) );
}


/*#
    @method show_all GtkWidget
    @brief Recursively shows a widget, and any child widgets (if the widget is a container).
 */
FALCON_FUNC Widget::show_all( VMARG )
{
    NO_ARGS
    gtk_widget_show_all( GET_WIDGET( vm->self() ) );
}


/*#
    @method hide_all GtkWidget
    @brief Recursively hides a widget and any child widgets.
 */
FALCON_FUNC Widget::hide_all( VMARG )
{
    NO_ARGS
    gtk_widget_hide_all( GET_WIDGET( vm->self() ) );
}


/*#
    @method map GtkWidget
    @brief This function is only for use in widget implementations.

    Causes a widget to be mapped if it isn't already.
 */
FALCON_FUNC Widget::map( VMARG )
{
    NO_ARGS
    gtk_widget_map( GET_WIDGET( vm->self() ) );
}


/*#
    @method unmap GtkWidget
    @brief This function is only for use in widget implementations.

    Causes a widget to be unmapped if it's currently mapped.
 */
FALCON_FUNC Widget::unmap( VMARG )
{
    NO_ARGS
    gtk_widget_unmap( GET_WIDGET( vm->self() ) );
}


/*#
    @method realize GtkWidget
    @brief Creates the GDK (windowing system) resources associated with a widget.

    For example, widget->window will be created when a widget is realized.
    Normally realization happens implicitly; if you show a widget and all its parent
    containers, then the widget will be realized and mapped automatically.

    Realizing a widget requires all the widget's parent widgets to be realized;
    calling gtk_widget_realize() realizes the widget's parents in addition to widget
    itself. If a widget is not yet inside a toplevel window when you realize it,
    bad things will happen.

    This function is primarily used in widget implementations, and isn't very
    useful otherwise. Many times when you think you might need it, a better
    approach is to connect to a signal that will be called after the widget is
    realized automatically, such as GtkWidget::expose-event.
    Or simply g_signal_connect() to the GtkWidget::realize signal.
 */
FALCON_FUNC Widget::realize( VMARG )
{
    NO_ARGS
    gtk_widget_realize( GET_WIDGET( vm->self() ) );
}


/*#
    @method unrealize GtkWidget
    @brief This function is only useful in widget implementations.

    Causes a widget to be unrealized (frees all GDK resources associated with
    the widget, such as widget->window).
 */
FALCON_FUNC Widget::unrealize( VMARG )
{
    NO_ARGS
    gtk_widget_unrealize( GET_WIDGET( vm->self() ) );
}


/*#
    @method queue_draw GtkWidget
    @brief Equivalent to calling widget.queue_draw_area() for the entire area of a widget.
 */
FALCON_FUNC Widget::queue_draw( VMARG )
{
    NO_ARGS
    gtk_widget_queue_draw( GET_WIDGET( vm->self() ) );
}


/*#
    @method queue_resize GtkWidget
    @brief This function is only for use in widget implementations.

    Flags a widget to have its size renegotiated; should be called when a widget
    for some reason has a new size request. For example, when you change the text
    in a GtkLabel, GtkLabel queues a resize to ensure there's enough space for
    the new text.
 */
FALCON_FUNC Widget::queue_resize( VMARG )
{
    NO_ARGS
    gtk_widget_queue_resize( GET_WIDGET( vm->self() ) );
}


/*#
    @method queue_resize_no_redraw GtkWidget
    @brief This function works like gtk_widget_queue_resize(), except that the widget is not invalidated.
*/
FALCON_FUNC Widget::queue_resize_no_redraw( VMARG )
{
    NO_ARGS
    gtk_widget_queue_resize_no_redraw( GET_WIDGET( vm->self() ) );
}


/*#
    @method size_request GtkWidget
    @brief Get the size requisition of the widget.
    @return GtkRequisition object

    Warning: gtk_widget_size_request has been deprecated since version 3.0 and
    should not be used in newly-written code.
    Use gtk_extended_layout_get_desired_size() instead.

    This function is typically used when implementing a GtkContainer subclass.
    Obtains the preferred size of a widget. The container uses this information
    to arrange its child widgets and decide what size allocations to give them
    with gtk_widget_size_allocate().

    You can also call this function from an application, with some caveats.
    Most notably, getting a size request requires the widget to be associated
    with a screen, because font information may be needed. Multihead-aware
    applications should keep this in mind.

    Also remember that the size request is not necessarily the size a widget
    will actually be allocated.
 */
FALCON_FUNC Widget::size_request( VMARG )
{
    NO_ARGS
    GtkRequisition req;
    gtk_widget_size_request( GET_WIDGET( vm->self() ), &req );
    vm->retval( new Gtk::Requisition( vm->findWKI( "GtkRequisition" )->asClass(), &req ) );
}


/*#
    @method get_child_requisition GtkWidget
    @brief This function is only for use in widget implementations.

    Warning: gtk_widget_get_child_requisition has been deprecated since version
    3.0 and should not be used in newly-written code.
    Use gtk_extended_layout_get_desired_size() instead.

    Obtains widget->requisition, unless someone has forced a particular geometry
    on the widget (e.g. with gtk_widget_set_size_request()), in which case it
    returns that geometry instead of the widget's requisition.

    This function differs from gtk_widget_size_request() in that it retrieves
    the last size request value from widget->requisition, while
    gtk_widget_size_request() actually calls the "size_request" method on widget
    to compute the size request and fill in widget->requisition, and only then
    returns widget->requisition.

    Because this function does not call the "size_request" method, it can only
    be used when you know that widget->requisition is up-to-date, that is,
    gtk_widget_size_request() has been called since the last time a resize was
    queued. In general, only container implementations have this information;
    applications should use gtk_widget_size_request().
 */
FALCON_FUNC Widget::get_child_requisition( VMARG )
{
    NO_ARGS
    GtkRequisition req;
    gtk_widget_get_child_requisition( GET_WIDGET( vm->self() ), &req );
    vm->retval( new Gtk::Requisition( vm->findWKI( "GtkRequisition" )->asClass(), &req ) );
}

#if 0 // todo
FALCON_FUNC Widget::size_allocate( VMARG );


/*#
    @method add_accelerator GtkWidget
    @brief Installs an accelerator for this widget in accel_group that causes accel_signal to be emitted if the accelerator is activated.
    @param accel_signal widget signal to emit on accelerator activation
    @param accel_group accel group for this widget, added to its toplevel (GtkAccelGroup).
    @param accel_key GDK keyval of the accelerator.
    @param accel_mods modifier key combination of the accelerator (GdkModifierType).
    @param accel_flags flag accelerators, e.g. GTK_ACCEL_VISIBLE (GtkAccelFlags).

    The accel_group needs to be added to the widget's toplevel via
    gtk_window_add_accel_group(), and the signal must be of type G_RUN_ACTION.
    Accelerators added through this function are not user changeable during runtime.
    If you want to support accelerators that can be changed by the user, use
    gtk_accel_map_add_entry() and gtk_widget_set_accel_path() or
    gtk_menu_item_set_accel_path() instead.
 */
FALCON_FUNC Widget::add_accelerator( VMARG )
{
    Item* i_sig = vm->param( 0 );
    Item* i_grp = vm->param( 1 );
    Item* i_key = vm->param( 2 );
    Item* i_mod = vm->param( 3 );
    Item* i_flg = vm->param( 4 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_sig || !i_sig->isString()
        || !i_grp || !i_grp->isObject() || !IS_DERIVED( i_grp, GtkAccelGroup )
        || !i_key || !i_key->isInteger()
        || !i_mod || !i_mod->isInteger()
        || !i_flg || !i_flg->isInteger() )
        throw_inv_params( "S,GtkAccelGroup,I,GdkModifierType,GtkAccelFlags" );
#endif
    AutoCString sig( i_sig->asString() );
    GtkAccelGroup* grp = GET_ACCELGROUP( *i_grp );
    gtk_widget_add_accelerator( GET_WIDGET( vm->self() ),
                                sig.c_str(),
                                grp,
                                i_key->asInteger(),
                                (GdkModifierType) i_mod->asInteger(),
                                (GtkAccelFlags) i_flg->asInteger() );
}


FALCON_FUNC Widget::remove_accelerator( VMARG );

FALCON_FUNC Widget::set_accel_path( VMARG );

FALCON_FUNC Widget::list_accel_closures( VMARG );
#endif


/*#
    @method can_activate_accel GtkWidget
    @brief Determines whether an accelerator that activates the signal identified by signal_id can currently be activated.
    @param signal_id the ID of a signal installed on widget
    @return TRUE if the accelerator can be activated.

    This is done by emitting the "can-activate-accel" signal on widget; if the
    signal isn't overridden by a handler or in a derived widget, then the default
    check is that the widget must be sensitive, and the widget and all its
    ancestors mapped.
 */
FALCON_FUNC Widget::can_activate_accel( VMARG )
{
    Item* i_id = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_id || !i_id->isInteger() )
        throw_inv_params( "I" );
#endif
    vm->retval( (bool) gtk_widget_can_activate_accel( GET_WIDGET( vm->self() ),
                                                      i_id->asInteger() ) );
}


/*#
    @method event GtkWidget
    @brief This function is used to emit the event signals on a widget (those signals should never be emitted without using this function to do so).
    @param event a GdkEvent
    @return return from the event signal emission (TRUE if the event was handled)

    Rarely-used function.

    If you want to synthesize an event though, don't use this function; instead,
    use gtk_main_do_event() so the event will behave as if it were in the event
    queue. Don't synthesize expose events; instead, use
    gdk_window_invalidate_rect() to invalidate a region of the window.
 */
FALCON_FUNC Widget::event( VMARG )
{
    Item* i_ev = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_ev || !i_ev->isObject() || !IS_DERIVED( i_ev, GdkEvent ) )
        throw_inv_params( "GdkEvent" );
#endif
    vm->retval( (bool) gtk_widget_event( GET_WIDGET( vm->self() ),
                                         GET_EVENT( *i_ev ) ) );
}


/*#
    @method activate GtkWidget
    @brief For widgets that can be "activated" (buttons, menu items, etc.) this function activates them.
    @return TRUE if the widget was activatable

    Activation is what happens when you press Enter on a widget during key
    navigation. If widget isn't activatable, the function returns FALSE.
 */
FALCON_FUNC Widget::activate( VMARG )
{
    NO_ARGS
    vm->retval( (bool) gtk_widget_activate( GET_WIDGET( vm->self() ) ) );
}


/*#
    @method reparent GtkWidget
    @brief Moves a widget from one container to another.
    @param new_parent a GtkContainer to move the widget into
 */
FALCON_FUNC Widget::reparent( VMARG )
{
    Item* i_wdt = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_wdt || !i_wdt->isObject() || !IS_DERIVED( i_wdt, GtkWidget ) )
        throw_inv_params( "GtkWidget" );
#endif
    gtk_widget_reparent( GET_WIDGET( vm->self() ),
                         GET_WIDGET( *i_wdt ) );
}


/*#
    @method intersect GtkWidget
    @brief Computes the intersection of a widget's area and area given as parameter.
    @param area a GdkRectangle
    @return a GdkRectangle intersection of the widget and area, or nil if there was no intersection
 */
FALCON_FUNC Widget::intersect( VMARG )
{
    Item* i_area = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_area || !i_area->isObject() || !IS_DERIVED( i_area, GdkRectangle ) )
        throw_inv_params( "GdkRectangle" );
#endif
    GdkRectangle res;
    if ( gtk_widget_intersect( GET_WIDGET( vm->self() ),
                               GET_RECTANGLE( *i_area ),
                               &res ) )
        vm->retval( new Gdk::Rectangle( vm->findWKI( "GdkRectangle" )->asClass(), &res ) );
    else
        vm->retnil();
}


/*#
    @method is_focus GtkWidget
    @brief Determines if the widget is the focus widget within its toplevel.
    @return TRUE if the widget is the focus widget.

    This does not mean that the HAS_FOCUS flag is necessarily set;
    HAS_FOCUS will only be set if the toplevel widget additionally has the
    global input focus.
 */
FALCON_FUNC Widget::is_focus( VMARG )
{
    NO_ARGS
    vm->retval( (bool) gtk_widget_is_focus( GET_WIDGET( vm->self() ) ) );
}


/*#
    @method grab_focus GtkWidget
    @brief Causes widget to have the keyboard focus for the GtkWindow it's inside.

    The widget must be a focusable widget, such as a GtkEntry; something like
    GtkFrame won't work.

    More precisely, it must have the GTK_CAN_FOCUS flag set.
    Use gtk_widget_set_can_focus() to modify that flag.
 */
FALCON_FUNC Widget::grab_focus( VMARG )
{
    NO_ARGS
    gtk_widget_grab_focus( GET_WIDGET( vm->self() ) );
}


/*#
    @method grab_default GtkWidget
    @brief Causes widget to become the default widget.

    The widget must have the GTK_CAN_DEFAULT flag set; typically you have to set this flag
    yourself by calling gtk_widget_set_can_default (widget, TRUE). The default widget
    is activated when the user presses Enter in a window. Default widgets must be
    activatable, that is, gtk_widget_activate() should affect them.
 */
FALCON_FUNC Widget::grab_default( VMARG )
{
    NO_ARGS
    gtk_widget_grab_default( GET_WIDGET( vm->self() ) );
}


/*#
    @method set_name GtkWidget
    @brief Attribute a name to the widget.
    @param name name for the widget

    Widgets can be named, which allows you to refer to them from a gtkrc file.
    You can apply a style to widgets with a particular name in the gtkrc file.
    See the documentation for gtkrc files (on the same page as the docs for GtkRcStyle).

    Note that widget names are separated by periods in paths (see gtk_widget_path()),
    so names with embedded periods may cause confusion.
 */
FALCON_FUNC Widget::set_name( VMARG )
{
    Item* i_name = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_name || !i_name->isString() )
        throw_inv_params( "S" );
#endif
    AutoCString name( i_name->asString() );
    gtk_widget_set_name( GET_WIDGET( vm->self() ), name.c_str() );
}


/*#
    @method get_name GtkWidget
    @brief Get the name of the widget.
    @return name of the widget.

    See GtkWidget.set_name() for the significance of widget names.
 */
FALCON_FUNC Widget::get_name( VMARG )
{
    NO_ARGS
    vm->retval( UTF8String( gtk_widget_get_name( GET_WIDGET( vm->self() ) ) ) );
}


/*#
    @method set_state GtkWidget
    @brief This function is for use in widget implementations.
    @param state new state for widget (GtkStateType)

    Sets the state of a widget (insensitive, prelighted, etc.) Usually you should
    set the state using wrapper functions such as gtk_widget_set_sensitive().
 */
FALCON_FUNC Widget::set_state( VMARG )
{
    Item* i_st = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_st || !i_st->isInteger() )
        throw_inv_params( "GtkStateType" );
#endif
    gtk_widget_set_state( GET_WIDGET( vm->self() ), (GtkStateType) i_st->asInteger() );
}


/*#
    @method set_sensitive GtkWidget
    @brief Sets the sensitivity of a widget.
    @param sensitive TRUE to make the widget sensitive

    A widget is sensitive if the user can interact with it.
    Insensitive widgets are "grayed out" and the user can't interact with them.
    Insensitive widgets are known as "inactive", "disabled", or "ghosted" in
    some other toolkits.
 */
FALCON_FUNC Widget::set_sensitive( VMARG )
{
    Item* i_sens = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_sens || !i_sens->isBoolean() )
        throw_inv_params( "B" );
#endif
    gtk_widget_set_sensitive( GET_WIDGET( vm->self() ), (gboolean) i_sens->asBoolean() );
}


/*#
    @method set_parent GtkWidget
    @brief This function is useful only when implementing subclasses of GtkContainer.
    @param parent parent container

    Sets the container as the parent of widget, and takes care of some deta
    ils such as updating the state and style of the child to reflect its new
    location. The opposite function is gtk_widget_unparent().
 */
FALCON_FUNC Widget::set_parent( VMARG )
{
    Item* i_par = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_par || !i_par->isObject() || !IS_DERIVED( i_par, GtkWidget ) )
        throw_inv_params( "GtkWidget" );
#endif
    gtk_widget_set_parent( GET_WIDGET( vm->self() ), GET_WIDGET( *i_par ) );
}


/*#
    @method set_parent_window GtkWidget
    @brief Sets a non default parent window for widget.
    @param parent_window the new parent window (GdkWindow).
 */
FALCON_FUNC Widget::set_parent_window( VMARG )
{
    Item* i_par = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_par || !i_par->isObject() || !IS_DERIVED( i_par, GdkWindow ) )
        throw_inv_params( "GdkWindow" );
#endif
    gtk_widget_set_parent_window( GET_WIDGET( vm->self() ), GET_GDKWINDOW( *i_par ) );
}


/*#
    @method get_parent_window GtkWidget
    @brief Gets widget's parent window.
    @return the parent window of widget (GdkWindow).
 */
FALCON_FUNC Widget::get_parent_window( VMARG )
{
    NO_ARGS
    vm->retval( new Gdk::Window( vm->findWKI( "GdkWindow" )->asClass(),
                    gtk_widget_get_parent_window( GET_WIDGET( vm->self() ) ) ) );
}


/*#
    @method set_events GtkWidget
    @brief Sets the event mask (see GdkEventMask) for a widget.
    @param an event mask, see GdkEventMask

    The event mask determines which events a widget will receive. Keep in mind that
    different widgets have different default event masks, and by changing the event
    mask you may disrupt a widget's functionality, so be careful. This function must
    be called while a widget is unrealized. Consider add_events() for
    widgets that are already realized, or if you want to preserve the existing event
    mask. This function can't be used with GTK_NO_WINDOW widgets; to get events on
    those widgets, place them inside a GtkEventBox and receive events on the event box.
 */
FALCON_FUNC Widget::set_events( VMARG )
{
    Item* i_ev = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_ev || !i_ev->isInteger() )
        throw_inv_params( "GdkEventMask" );
#endif
    gtk_widget_set_events( GET_WIDGET( vm->self() ), i_ev->asInteger() );
}


/*#
    @method get_events GtkWidget
    @brief Returns the event mask for the widget.
    @return event mask for widget (GdkEventMask)

    (A bitfield containing flags from the GdkEventMask enumeration.)
    These are the events that the widget will receive.
 */
FALCON_FUNC Widget::get_events( VMARG )
{
    NO_ARGS
    vm->retval( gtk_widget_get_events( GET_WIDGET( vm->self() ) ) );
}


/*#
    @method add_events GtkWidget
    @brief Adds the events in the bitfield events to the event mask for widget.
    @param events an event mask, see GdkEventMask
 */
FALCON_FUNC Widget::add_events( VMARG )
{
    Item* i_ev = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_ev || !i_ev->isInteger() )
        throw_inv_params( "GdkEventMask" );
#endif
    gtk_widget_add_events( GET_WIDGET( vm->self() ), i_ev->asInteger() );
}


/*#
    @method set_extension_events GtkWidget
    @brief Sets the extension events mask to mode.
    @param mode bitfield of extension events to receive (GdkExtensionMode).

    See GdkExtensionMode and gdk_input_set_extension_events().
 */
FALCON_FUNC Widget::set_extension_events( VMARG )
{
    Item* i_mode = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_mode || !i_mode->isInteger() )
        throw_inv_params( "GdkExtensionMode" );
#endif
    gtk_widget_set_extension_events( GET_WIDGET( vm->self() ),
                                     (GdkExtensionMode) i_mode->asInteger() );
}


/*#
    @method get_extension_events GtkWidget
    @brief Retrieves the extension events the widget will receive.
    @return extension events for widget (GdkExtensionMode).

    See gdk_input_set_extension_events().
 */
FALCON_FUNC Widget::get_extension_events( VMARG )
{
    NO_ARGS
    vm->retval( (int64) gtk_widget_get_extension_events( GET_WIDGET( vm->self() ) ) );
}


#if GTK_CHECK_VERSION( 3, 0, 0 )
FALCON_FUNC Widget::set_device_events( VMARG );
FALCON_FUNC Widget::get_device_events( VMARG );
FALCON_FUNC Widget::add_device_events( VMARG );
#endif


/*#
    @method get_toplevel GtkWidget
    @brief This function returns the topmost widget in the container hierarchy widget is a part of.
    @return the topmost ancestor of widget, or widget itself if there's no ancestor.

    If widget has no parent widgets, it will be returned as the topmost widget.

    Note the difference in behavior vs. gtk_widget_get_ancestor();
    gtk_widget_get_ancestor (widget, GTK_TYPE_WINDOW) would return NULL if widget
    wasn't inside a toplevel window, and if the window was inside a GtkWindow-derived
    widget which was in turn inside the toplevel GtkWindow. While the second case
    may seem unlikely, it actually happens when a GtkPlug is embedded inside a
    GtkSocket within the same application.

    To reliably find the toplevel GtkWindow, use gtk_widget_get_toplevel() and
    check if the TOPLEVEL flags is set on the result.
 */
FALCON_FUNC Widget::get_toplevel( VMARG )
{
    NO_ARGS
    vm->retval( new Widget( vm->findWKI( "GtkWidget" )->asClass(),
                            gtk_widget_get_toplevel( GET_WIDGET( vm->self() ) ) ) );
}


/*#
    @method get_ancestor GtkWidget
    @brief Gets the first ancestor of widget with type widget_type.
    @param widget_type ancestor type (GType)
    @return the ancestor widget, or NULL if not found.

    For example, gtk_widget_get_ancestor (widget, GTK_TYPE_BOX) gets the first
    GtkBox that's an ancestor of widget. See note about checking for a toplevel
    GtkWindow in the docs for gtk_widget_get_toplevel().

    Note that unlike gtk_widget_is_ancestor(), gtk_widget_get_ancestor() considers
    widget to be an ancestor of itself.
 */
FALCON_FUNC Widget::get_ancestor( VMARG )
{
    Item* i_tp = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_tp || !i_tp->isInteger() )
        throw_inv_params( "GType" );
#endif
    GtkWidget* wdt = gtk_widget_get_ancestor( GET_WIDGET( vm->self() ),
                                              (GType) i_tp->asInteger() );
    if ( wdt )
        vm->retval( new Gtk::Widget( vm->findWKI( "GtkWidget" )->asClass(), wdt ) );
    else
        vm->retnil();
}


/*#
    @method get_colormap GtkWidget
    @brief Gets the colormap that will be used to render widget.
    @return the colormap used by widget (GdkColormap).
 */
FALCON_FUNC Widget::get_colormap( VMARG )
{
    NO_ARGS
    vm->retval( new Gdk::Colormap( vm->findWKI( "GdkColormap" )->asClass(),
                        gtk_widget_get_colormap( GET_WIDGET( vm->self() ) ) ) );
}


/*#
    @method set_colormap GtkWidget
    @brief Sets the colormap for the widget to the given value.
    @param colormap a GdkColormap

    Widget must not have been previously realized. This probably should only be
    used from an init() function (i.e. from the constructor for the widget).
 */
FALCON_FUNC Widget::set_colormap( VMARG )
{
    Item* i_map = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_map || !i_map->isObject() || !IS_DERIVED( i_map, GdkColormap ) )
        throw_inv_params( "GdkColormap" );
#endif
    gtk_widget_set_colormap( GET_WIDGET( vm->self() ),
                             GET_COLORMAP( *i_map ) );
}


/*#
    @method get_visual GtkWidget
    @brief Gets the visual that will be used to render widget.
    @return the GdkVisual for widget.
 */
FALCON_FUNC Widget::get_visual( VMARG )
{
    NO_ARGS
    vm->retval( new Gdk::Visual( vm->findWKI( "GdkVisual" )->asClass(),
                        gtk_widget_get_visual( GET_WIDGET( vm->self() ) ) ) );
}


/*#
    @method get_pointer GtkWidget
    @brief Obtains the location of the mouse pointer in widget coordinates.
    @return An array [ x coordinate, y coordinate ]

    Widget coordinates are a bit odd; for historical reasons, they are defined as
    widget->window coordinates for widgets that are not GTK_NO_WINDOW widgets,
    and are relative to widget->allocation.x, widget->allocation.y for widgets
    that are GTK_NO_WINDOW widgets.

 */
FALCON_FUNC Widget::get_pointer( VMARG )
{
    NO_ARGS
    gint x, y;
    gtk_widget_get_pointer( GET_WIDGET( vm->self() ), &x, &y );
    CoreArray* arr = new CoreArray( 2 );
    arr->append( x );
    arr->append( y );
    vm->retval( arr );
}


/*#
    @method is_ancestor GtkWidget
    @brief Determines whether widget is somewhere inside ancestor, possibly with intermediate containers.
    @param ancestor another GtkWidget
    @return TRUE if ancestor contains widget as a child, grandchild, great grandchild, etc.
 */
FALCON_FUNC Widget::is_ancestor( VMARG )
{
    Item* i_wdt = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_wdt || !IS_DERIVED( i_wdt, GtkWidget ) )
        throw_inv_params( "GtkWidget" );
#endif
    vm->retval( (bool) gtk_widget_is_ancestor( GET_WIDGET( vm->self() ),
                                               GET_WIDGET( *i_wdt ) ) );
}


/*#
    @method translate_coordinates GtkWidget
    @brief Translate coordinates relative to src_widget's allocation to coordinates relative to dest_widget's allocations.
    @param dest_widget a GtkWidget
    @param src_x X position relative to src_widget
    @param src_y Y position relative to src_widget
    @return An array [ X position, Y position ] relative to dest_widget, or nil if either widget was not realized, or there was no common ancestor.

    In order to perform this operation, both widgets must be realized, and must
    share a common toplevel.
 */
FALCON_FUNC Widget::translate_coordinates( VMARG )
{
    Item* i_wdt = vm->param( 0 );
    Item* i_x = vm->param( 1 );
    Item* i_y = vm->param( 2 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_wdt || !i_wdt->isObject() || !IS_DERIVED( i_wdt, gtkWidget )
        || !i_x || !i_x->isInteger()
        || !i_y || !i_y->isInteger() )
        throw_inv_params( "GtkWidget,I,I" );
#endif
    gint x, y;
    if ( gtk_widget_translate_coordinates( GET_WIDGET( vm->self() ),
                                           GET_WIDGET( *i_wdt ),
                                           i_x->asInteger(),
                                           i_y->asInteger(),
                                           &x, &y ) )
    {
        CoreArray* arr = new CoreArray( 2 );
        arr->append( x );
        arr->append( y );
        vm->retval( arr );
    }
    else
        vm->retnil();
}



/*#
    @method hide_on_delete GtkWidget
    @brief Utility function.
    @return always true

    Intended to be connected to the "delete-event" signal on a GtkWindow.
    The function calls gtk_widget_hide() on its argument, then returns TRUE.
    If connected to ::delete-event, the result is that clicking the close button
    for a window (on the window frame, top right corner usually) will hide but
    not destroy the window.

    By default, GTK+ destroys windows when ::delete-event is received.
 */
FALCON_FUNC Widget::hide_on_delete( VMARG )
{
    NO_ARGS
    vm->retval( (bool) gtk_widget_hide_on_delete( GET_WIDGET( vm->self() ) ) );
}


#if 0 // todo
FALCON_FUNC Widget::set_style( VMARG );
#endif


/*#
    @method ensure_style GtkWidget
    @brief Ensures that widget has a style (widget->style).

    Not a very useful function; most of the time, if you want the style, the
    widget is realized, and realized widgets are guaranteed to have a style already.
 */
FALCON_FUNC Widget::ensure_style( VMARG )
{
    NO_ARGS
    gtk_widget_ensure_style( GET_WIDGET( vm->self() ) );
}


#if 0 // todo
FALCON_FUNC Widget::get_style( VMARG );
#endif


/*#
    @method reset_rc_styles GtkWidget
    @brief Reset the styles of widget and all descendents, so when they are looked up again, they get the correct values for the currently loaded RC file settings.

    This function is not useful for applications.
 */
FALCON_FUNC Widget::reset_rc_styles( VMARG )
{
    NO_ARGS
    gtk_widget_reset_rc_styles( GET_WIDGET( vm->self() ) );
}


/*#
    @method push_colormap GtkWidget
    @brief Pushes cmap onto a global stack of colormaps; the topmost colormap on the stack will be used to create all widgets.
    @param cmap a GdkColormap

    Remove cmap with gtk_widget_pop_colormap(). There's little reason to use this function.
 */
FALCON_FUNC Widget::push_colormap( VMARG )
{
    Item* i_map = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_map || !i_map->isObject() || !IS_DERIVED( i_map, GdkColormap ) )
        throw_inv_params( "GdkColormap" );
#endif
    gtk_widget_push_colormap( GET_COLORMAP( *i_map ) );
}


/*#
    @method pop_colormap GtkWidget
    @brief Removes a colormap pushed with gtk_widget_push_colormap().
 */
FALCON_FUNC Widget::pop_colormap( VMARG )
{
    NO_ARGS
    gtk_widget_pop_colormap();
}


/*#
    @method set_default_colormap GtkWidget
    @brief Sets the default colormap to use when creating widgets.
    @param colormap a GdkColormap

    gtk_widget_push_colormap() is a better function to use if you only want to
    affect a few widgets, rather than all widgets.
 */
FALCON_FUNC Widget::set_default_colormap( VMARG )
{
    Item* i_map = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_map || !i_map->isObject() || !IS_DERIVED( i_map, GdkColormap ) )
        throw_inv_params( "GdkColormap" );
#endif
    gtk_widget_set_default_colormap( GET_COLORMAP( *i_map ) );
}


#if 0 // todo
FALCON_FUNC Widget::get_default_style( VMARG );
#endif


/*#
    @method get_default_colormap GtkWidget
    @brief Obtains the default colormap used to create widgets.
    @return default widget colormap.
 */
FALCON_FUNC Widget::get_default_colormap( VMARG )
{
    NO_ARGS
    vm->retval( new Gdk::Colormap( vm->findWKI( "GdkColormap" )->asClass(),
                                   gtk_widget_get_default_colormap() ) );
}


/*#
    @method get_default_visual GtkWidget
    @brief Obtains the visual of the default colormap.
    @return visual of the default colormap.

    Not really useful; used to be useful before gdk_colormap_get_visual() existed.
 */
FALCON_FUNC Widget::get_default_visual( VMARG )
{
    NO_ARGS
    vm->retval( new Gdk::Visual( vm->findWKI( "GdkVisual" )->asClass(),
                                 gtk_widget_get_default_visual() ) );
}


/*#
    @method set_direction GtkWidget
    @brief Sets the reading direction on a particular widget.
    @param dir the new direction (GtkTextDirection).

    This direction controls the primary direction for widgets containing text,
    and also the direction in which the children of a container are packed. The
    ability to set the direction is present in order so that correct localization
    into languages with right-to-left reading directions can be done. Generally,
    applications will let the default reading direction present, except for
    containers where the containers are arranged in an order that is explicitely
    visual rather than logical (such as buttons for text justification).

    If the direction is set to GTK_TEXT_DIR_NONE, then the value set by
    gtk_widget_set_default_direction() will be used.
 */
FALCON_FUNC Widget::set_direction( VMARG )
{
    Item* i_dir = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_dir || !i_dir->isInteger() )
        throw_inv_params( "GtkTextDirection" );
#endif
    gtk_widget_set_direction( GET_WIDGET( vm->self() ),
                              (GtkTextDirection) i_dir->asInteger() );
}


/*#
    @method get_direction GtkWidget
    @brief Gets the reading direction for a particular widget.
    @return the reading direction for the widget (GtkTextDirection).
 */
FALCON_FUNC Widget::get_direction( VMARG )
{
    NO_ARGS
    vm->retval( (int64) gtk_widget_get_direction( GET_WIDGET( vm->self() ) ) );
}


/*#
    @method set_default_direction GtkWidget
    @brief Sets the default reading direction for widgets where the direction has not been explicitly set by gtk_widget_set_direction().
    @param dir the new default direction (GtkTextDirection). This cannot be GTK_TEXT_DIR_NONE.
 */
FALCON_FUNC Widget::set_default_direction( VMARG )
{
    Item* i_dir = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_dir || !i_dir->isInteger() )
        throw_inv_params( "GtkTextDirection" );
#endif
    gtk_widget_set_default_direction( (GtkTextDirection) i_dir->asInteger() );
}


/*#
    @method get_default_direction GtkWidget
    @brief Obtains the current default reading direction.
    @return the current default direction (GtkTextDirection).
 */
FALCON_FUNC Widget::get_default_direction( VMARG )
{
    NO_ARGS
    vm->retval( (int64) gtk_widget_get_default_direction() );
}


/*#
    @method shape_combine_mask GtkWidget
    @brief Sets a shape for this widget's GDK window.
    @param shape_mask shape to be added, or NULL to remove an existing shape (GdkBitmap).
    @param offset_x X position of shape mask with respect to window
    @param offset_y Y position of shape mask with respect to window

    This allows for transparent windows etc., see gdk_window_shape_combine_mask()
    for more information.
 */
FALCON_FUNC Widget::shape_combine_mask( VMARG )
{
    Item* i_msk = vm->param( 0 );
    Item* i_x = vm->param( 1 );
    Item* i_y = vm->param( 2 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_msk || !i_msk->isObject() || !IS_DERIVED( i_msk, GdkBitmap )
        || !i_x || !i_x->isInteger()
        || !i_y || !i_y->isInteger() )
        throw_inv_params( "GdkBitmap,I,I" );
#endif
    gtk_widget_shape_combine_mask( GET_WIDGET( vm->self() ),
                                   GET_BITMAP( *i_msk ),
                                   i_x->asInteger(),
                                   i_y->asInteger() );
}


/*#
    @method input_shape_combine_mask GtkWidget
    @brief Sets an input shape for this widget's GDK window.
    @param shape_mask shape to be added, or NULL to remove an existing shape (GdkBitmap).
    @param offset_x X position of shape mask with respect to window
    @param offset_y Y position of shape mask with respect to window

    This allows for windows which react to mouse click in a nonrectangular
    region, see gdk_window_input_shape_combine_mask() for more information.
 */
FALCON_FUNC Widget::input_shape_combine_mask( VMARG )
{
    Item* i_msk = vm->param( 0 );
    Item* i_x = vm->param( 1 );
    Item* i_y = vm->param( 2 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_msk || !i_msk->isObject() || !IS_DERIVED( i_msk, GdkBitmap )
        || !i_x || !i_x->isInteger()
        || !i_y || !i_y->isInteger() )
        throw_inv_params( "GdkBitmap,I,I" );
#endif
    gtk_widget_input_shape_combine_mask( GET_WIDGET( vm->self() ),
                                         GET_BITMAP( *i_msk ),
                                         i_x->asInteger(),
                                         i_y->asInteger() );
}


/*#
    @method path GtkWidget
    @brief Obtains the full path to widget.
    @param reverse TRUE to get the path in reverse order
    @return the path

    The path is simply the name of a widget and all its parents in the container
    hierarchy, separated by periods. The name of a widget comes from
    gtk_widget_get_name(). Paths are used to apply styles to a widget in gtkrc
    configuration files. Widget names are the type of the widget by default
    (e.g. "GtkButton") or can be set to an application-specific value with
    gtk_widget_set_name(). By setting the name of a widget, you allow users or
    theme authors to apply styles to that specific widget in their gtkrc file.

    Setting reverse to TRUE returns the path in reverse order, i.e. starting with
    widget's name instead of starting with the name of widget's outermost ancestor.
 */
FALCON_FUNC Widget::path( VMARG )
{
    Item* i_rev = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_rev || !i_rev->isBoolean() )
        throw_inv_params( "B" );
#endif
    gchar* path;
    if ( i_rev->asBoolean() )
        gtk_widget_path( GET_WIDGET( vm->self() ), NULL, NULL, &path );
    else
        gtk_widget_path( GET_WIDGET( vm->self() ), NULL, &path, NULL );
    vm->retval( UTF8String( path ) );
    g_free( path );
}


/*#
    @method class_path GtkWidget
    @brief Same as gtk_widget_path(), but always uses the name of a widget's type, never uses a custom name set with gtk_widget_set_name().
    @param reverse TRUE to get the path in reverse order
    @return the class path
 */
FALCON_FUNC Widget::class_path( VMARG )
{
    Item* i_rev = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_rev || !i_rev->isBoolean() )
        throw_inv_params( "B" );
#endif
    gchar* path;
    if ( i_rev->asBoolean() )
        gtk_widget_class_path( GET_WIDGET( vm->self() ), NULL, NULL, &path );
    else
        gtk_widget_class_path( GET_WIDGET( vm->self() ), NULL, &path, NULL );
    vm->retval( UTF8String( path ) );
    g_free( path );
}


/*#
    @method get_composite_name GtkWidget
    @brief Obtains the composite name of a widget.
    @return the composite name of widget, or NULL if widget is not a composite child.
 */
FALCON_FUNC Widget::get_composite_name( VMARG )
{
    NO_ARGS
    gchar* name = gtk_widget_get_composite_name( GET_WIDGET( vm->self() ) );
    if ( name )
    {
        vm->retval( UTF8String( name ) );
        g_free( name );
    }
    else
        vm->retnil();
}


#if 0 // todo
//FALCON_FUNC Widget::modify_style( VMARG );
//FALCON_FUNC Widget::get_modifier_style( VMARG );
#endif


/*#
    @method modify_fg GtkWidget
    @brief Sets the foreground color for a widget in a particular state.
    @param state the state for which to set the foreground color (GtkStateType).
    @param color the color to assign, or NULL to undo the effect of previous calls to of gtk_widget_modify_fg()

    All other style values are left untouched. See also gtk_widget_modify_style().
 */
FALCON_FUNC Widget::modify_fg( VMARG )
{
    Item* i_st = vm->param( 0 );
    Item* i_clr = vm->param( 1 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_st || !i_st->isInteger()
        || !i_clr || !( i_clr->isNil() || ( i_clr->isObject()
        && IS_DERIVED( i_clr, GdkColor ) ) ) )
        throw_inv_params( "GtkStateType,[GdkColor]" );
#endif
    gtk_widget_modify_fg( GET_WIDGET( vm->self() ),
                          (GtkStateType) i_st->asInteger(),
                          i_clr->isNil() ? NULL : GET_COLOR( *i_clr ) );
}


/*#
    @method modify_bg GtkWidget
    @brief Sets the background color for a widget in a particular state.
    @param state the state for which to set the background color (GtkStateType).
    @param color the color to assign, or NULL to undo the effect of previous calls to of gtk_widget_modify_bg().

    All other style values are left untouched. See also gtk_widget_modify_style().

    Note that "no window" widgets (which have the GTK_NO_WINDOW flag set) draw on
    their parent container's window and thus may not draw any background themselves.
    This is the case for e.g. GtkLabel. To modify the background of such widgets,
    you have to set the background color on their parent; if you want to set the
    background of a rectangular area around a label, try placing the label in a
    GtkEventBox widget and setting the background color on that.
 */
FALCON_FUNC Widget::modify_bg( VMARG )
{
    Item* i_st = vm->param( 0 );
    Item* i_clr = vm->param( 1 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_st || !i_st->isInteger()
        || !i_clr || !( i_clr->isNil() || ( i_clr->isObject()
        && IS_DERIVED( i_clr, GdkColor ) ) ) )
        throw_inv_params( "GtkStateType,[GdkColor]" );
#endif
    gtk_widget_modify_bg( GET_WIDGET( vm->self() ),
                          (GtkStateType) i_st->asInteger(),
                          i_clr->isNil() ? NULL : GET_COLOR( *i_clr ) );
}


/*#
    @method modify_text GtkWidget
    @brief Sets the text color for a widget in a particular state.
    @param state the state for which to set the text color (GtkStateType).
    @param color the color to assign, or NULL to undo the effect of previous calls to of gtk_widget_modify_text().

    All other style values are left untouched. The text color is the foreground
    color used along with the base color (see gtk_widget_modify_base()) for
    widgets such as GtkEntry and GtkTextView. See also gtk_widget_modify_style().
 */
FALCON_FUNC Widget::modify_text( VMARG )
{
    Item* i_st = vm->param( 0 );
    Item* i_clr = vm->param( 1 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_st || !i_st->isInteger()
        || !i_clr || !( i_clr->isNil() || ( i_clr->isObject()
        && IS_DERIVED( i_clr, GdkColor ) ) ) )
        throw_inv_params( "GtkStateType,[GdkColor]" );
#endif
    gtk_widget_modify_text( GET_WIDGET( vm->self() ),
                            (GtkStateType) i_st->asInteger(),
                            i_clr->isNil() ? NULL : GET_COLOR( *i_clr ) );
}


/*#
    @method modify_base GtkWidget
    @brief Sets the base color for a widget in a particular state.
    @param state the state for which to set the base color (GtkStateType).
    @param the color to assign, or NULL to undo the effect of previous calls to of gtk_widget_modify_base().

    All other style values are left untouched. The base color is the background
    color used along with the text color (see gtk_widget_modify_text()) for
    widgets such as GtkEntry and GtkTextView. See also gtk_widget_modify_style().

    Note that "no window" widgets (which have the GTK_NO_WINDOW flag set) draw
    on their parent container's window and thus may not draw any background
    themselves. This is the case for e.g. GtkLabel. To modify the background of
    such widgets, you have to set the base color on their parent; if you want to
    set the background of a rectangular area around a label, try placing the
    label in a GtkEventBox widget and setting the base color on that.
 */
FALCON_FUNC Widget::modify_base( VMARG )
{
    Item* i_st = vm->param( 0 );
    Item* i_clr = vm->param( 1 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_st || !i_st->isInteger()
        || !i_clr || !( i_clr->isNil() || ( i_clr->isObject()
        && IS_DERIVED( i_clr, GdkColor ) ) ) )
        throw_inv_params( "GtkStateType,[GdkColor]" );
#endif
    gtk_widget_modify_base( GET_WIDGET( vm->self() ),
                            (GtkStateType) i_st->asInteger(),
                            i_clr->isNil() ? NULL : GET_COLOR( *i_clr ) );
}



//FALCON_FUNC Widget::modify_font( VMARG );

//FALCON_FUNC Widget::modify_cursor( VMARG );

//FALCON_FUNC Widget::create_pango_context( VMARG );

//FALCON_FUNC Widget::get_pango_context( VMARG );

//FALCON_FUNC Widget::create_pango_layout( VMARG );

//FALCON_FUNC Widget::widget_render_icon( VMARG );

//FALCON_FUNC Widget::pop_composite_child( VMARG );

//FALCON_FUNC Widget::push_composite_child( VMARG );

//FALCON_FUNC Widget::queue_clear( VMARG );

//FALCON_FUNC Widget::queue_clear_area( VMARG );

//FALCON_FUNC Widget::queue_draw_area( VMARG );

//FALCON_FUNC Widget::reset_shapes( VMARG );

//FALCON_FUNC Widget::set_app_paintable( VMARG );

//FALCON_FUNC Widget::set_double_buffered( VMARG );

//FALCON_FUNC Widget::set_redraw_on_allocate( VMARG );

//FALCON_FUNC Widget::set_composite_name( VMARG );

//FALCON_FUNC Widget::set_scroll_adjustments( VMARG );

//FALCON_FUNC Widget::mnemonic_activate( VMARG );

//FALCON_FUNC Widget::class_install_style_property( VMARG );

//FALCON_FUNC Widget::class_install_style_property_parser( VMARG );

//FALCON_FUNC Widget::class_find_style_property( VMARG );

//FALCON_FUNC Widget::class_list_style_properties( VMARG );

//FALCON_FUNC Widget::region_intersect( VMARG );

//FALCON_FUNC Widget::send_expose( VMARG );

//FALCON_FUNC Widget::style_get( VMARG );

//FALCON_FUNC Widget::style_get_property( VMARG );

//FALCON_FUNC Widget::style_get_valist( VMARG );

//FALCON_FUNC Widget::style_attach( VMARG );

//FALCON_FUNC Widget::get_accessible( VMARG );

//FALCON_FUNC Widget::child_focus( VMARG );

//FALCON_FUNC Widget::child_notify( VMARG );

//FALCON_FUNC Widget::freeze_child_notify( VMARG );

//FALCON_FUNC Widget::get_child_visible( VMARG );

//FALCON_FUNC Widget::get_parent( VMARG );

//FALCON_FUNC Widget::get_settings( VMARG );

//FALCON_FUNC Widget::get_clipboard( VMARG );

//FALCON_FUNC Widget::get_display( VMARG );

//FALCON_FUNC Widget::get_root_window( VMARG );

//FALCON_FUNC Widget::get_screen( VMARG );

//FALCON_FUNC Widget::has_screen( VMARG );


/*#
    @method get_size_request GtkWidget
    @brief Get the size requested for the widget
    @return [ width, height ]

    Gets the size request that was explicitly set for the widget using
    gtk_widget_set_size_request(). A value of -1 stored in width or height
    indicates that that dimension has not been set explicitly and the natural
    requisition of the widget will be used intead. See gtk_widget_set_size_request().

    To get the size a widget will actually use, call gtk_widget_size_request()
    instead of this function.
 */
FALCON_FUNC Widget::get_size_request( VMARG )
{
    NO_ARGS
    MYSELF;
    GET_OBJ( self );
    gint w, h;
    gtk_widget_get_size_request( (GtkWidget*)_obj, &w, &h );
    CoreArray* arr = new CoreArray;
    arr->append( w );
    arr->append( h );
    vm->retval( arr );
}


//FALCON_FUNC Widget::set_child_visible( VMARG );

//FALCON_FUNC Widget::set_default_visual( VMARG );


/*#
    @method set_size_request GtkWidget
    @brief Sets the minimum size of a widget
    @param width (integer)
    @param height (integer)

    that is, the widget's size request will be width by height.
    You can use this function to force a widget to be either larger or smaller
    than it normally would be.

    In most cases, gtk_window_set_default_size() is a better choice for toplevel
    windows than this function; setting the default size will still allow users
    to shrink the window. Setting the size request will force them to leave the
    window at least as large as the size request. When dealing with window sizes,
    gtk_window_set_geometry_hints() can be a useful function as well.

    Note the inherent danger of setting any fixed size - themes, translations
    into other languages, different fonts, and user action can all change the
    appropriate size for a given widget. So, it's basically impossible to hardcode
    a size that will always be correct.

    The size request of a widget is the smallest size a widget can accept while
    still functioning well and drawing itself correctly. However in some strange
    cases a widget may be allocated less than its requested size, and in many
    cases a widget may be allocated more space than it requested.

    If the size request in a given direction is -1 (unset), then the "natural"
    size request of the widget will be used instead.

    Widgets can't actually be allocated a size less than 1 by 1, but you can pass
    0,0 to this function to mean "as small as possible."
 */
FALCON_FUNC Widget::set_size_request( VMARG )
{
    Item* i_w = vm->param( 0 );
    Item* i_h = vm->param( 1 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_w || i_w->isNil() || !i_w->isInteger()
        || !i_h || i_h->isNil() || !i_h->isInteger() )
        throw_inv_params( "I,I" );
#endif
    MYSELF;
    GET_OBJ( self );
    gtk_widget_set_size_request( (GtkWidget*)_obj, i_w->asInteger(), i_h->asInteger() );
}


//FALCON_FUNC Widget::set_visual( VMARG );

//FALCON_FUNC Widget::thaw_child_notify( VMARG );

//FALCON_FUNC Widget::set_no_show_all( VMARG );

//FALCON_FUNC Widget::get_no_show_all( VMARG );

//FALCON_FUNC Widget::list_mnemonic_labels( VMARG );

//FALCON_FUNC Widget::add_mnemonic_label( VMARG );

//FALCON_FUNC Widget::remove_mnemonic_label( VMARG );

//FALCON_FUNC Widget::get_action( VMARG );

//FALCON_FUNC Widget::is_composited( VMARG );

//FALCON_FUNC Widget::error_bell( VMARG );

//FALCON_FUNC Widget::keynav_failed( VMARG );

//FALCON_FUNC Widget::get_tooltip_markup( VMARG );

//FALCON_FUNC Widget::set_tooltip_markup( VMARG );

//FALCON_FUNC Widget::get_tooltip_text( VMARG );

//FALCON_FUNC Widget::set_tooltip_text( VMARG );

//FALCON_FUNC Widget::get_tooltip_window( VMARG );

//FALCON_FUNC Widget::tooltip_window( VMARG );

//FALCON_FUNC Widget::get_has_tooltip( VMARG );

//FALCON_FUNC Widget::set_has_tooltip( VMARG );

//FALCON_FUNC Widget::trigger_tooltip_query( VMARG );

//FALCON_FUNC Widget::get_snapshot( VMARG );

//FALCON_FUNC Widget::get_window( VMARG );

//FALCON_FUNC Widget::get_allocation( VMARG );

//FALCON_FUNC Widget::set_allocation( VMARG );

//FALCON_FUNC Widget::get_app_paintable( VMARG );

//FALCON_FUNC Widget::get_can_default( VMARG );

//FALCON_FUNC Widget::set_can_default( VMARG );

//FALCON_FUNC Widget::get_can_focus( VMARG );

//FALCON_FUNC Widget::set_can_focus( VMARG );

//FALCON_FUNC Widget::get_double_buffered( VMARG );

//FALCON_FUNC Widget::get_has_window( VMARG );

//FALCON_FUNC Widget::set_has_window( VMARG );

//FALCON_FUNC Widget::get_sensitive( VMARG );

//FALCON_FUNC Widget::is_sensitive( VMARG );

//FALCON_FUNC Widget::get_state( VMARG );

//FALCON_FUNC Widget::get_visible( VMARG );

//FALCON_FUNC Widget::set_visible( VMARG );

//FALCON_FUNC Widget::has_default( VMARG );

//FALCON_FUNC Widget::has_focus( VMARG );

//FALCON_FUNC Widget::has_grab( VMARG );

//FALCON_FUNC Widget::has_rc_style( VMARG );

//FALCON_FUNC Widget::is_drawable( VMARG );

//FALCON_FUNC Widget::is_toplevel( VMARG );

//FALCON_FUNC Widget::set_window( VMARG );

//FALCON_FUNC Widget::set_receives_default( VMARG );

//FALCON_FUNC Widget::get_receives_default( VMARG );

//FALCON_FUNC Widget::set_realized( VMARG );

//FALCON_FUNC Widget::get_realized( VMARG );

//FALCON_FUNC Widget::set_mapped( VMARG );

//FALCON_FUNC Widget::get_mapped( VMARG );

//FALCON_FUNC Widget::get_requisition( VMARG );


} // Gtk
} // Falcon
