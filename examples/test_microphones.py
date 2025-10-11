#!/usr/bin/env python3
"""
Microphone Quality Testing Example

Automatically tests all available microphones and recommends the best one.
"""
from parakeet_stream import Parakeet, Microphone


def main():
    print("Microphone Quality Testing Example")
    print("=" * 60)

    # Initialize transcriber
    print("\nInitializing Parakeet transcriber...")
    pk = Parakeet()

    # Test all microphones
    print("\nStarting microphone quality test...")
    print("This will test each available microphone.")
    print("You'll be asked to read a sentence for each one.\n")

    results = Microphone.test_all(pk, duration=5.0, playback=False)

    if not results:
        print("\n❌ No microphones found or all tests failed.")
        return

    # Show how to use best microphone
    best = results[0]

    if best.has_audio:
        print("\n" + "=" * 60)
        print("NEXT STEPS")
        print("=" * 60)
        print(f"\n1. Use best microphone for live transcription:")
        print(f"   >>> from parakeet_stream import Parakeet, Microphone")
        print(f"   >>> pk = Parakeet()")
        print(f"   >>> mic = Microphone(device={best.microphone.device})")
        print(f"   >>> live = pk.listen(microphone=mic)")

        print(f"\n2. Replay any recording:")
        print(f"   >>> results[0].clip.play()  # Best mic")
        print(f"   >>> results[1].clip.play()  # Second best")

        print(f"\n3. Compare quality:")
        for i, result in enumerate(results[:3], 1):
            if result.has_audio:
                print(f"   Mic {i}: {result.microphone.name} - {result.quality_score:.1%}")
    else:
        print("\n⚠️ No working microphones detected.")
        print("Please check that:")
        print("  - Microphones are properly connected")
        print("  - Microphones are not muted")
        print("  - You have granted microphone permissions")


if __name__ == "__main__":
    main()
